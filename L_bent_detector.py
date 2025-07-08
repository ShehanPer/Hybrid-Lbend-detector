import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import time
import math

class CombinedDetector:
    def __init__(self,template_path, model_path,threshold = 0.55,iou_threshold = 0.3,confidence_threshold = 0.5):

        #set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device : {self.device}")

        # Load the template image
        template_bgr = cv2.imread(template_path)
        template_bgr = cv2.resize(template_bgr,(64,64))
        self.template_tensor = self.to_tensor_from_red_channel(template_bgr).to(self.device)

        # Genrate rotated templates
        angles = [0,45,90,135,180,225,270,315]
        self.rotated_templates = self.get_rotated_templates(self.template_tensor, angles).to(self.device)

        # Load the model
        try:
            self.model = torch.jit.load(model_path, map_location=self.device)
            print("Model loaded successfully.")

        except:
            print("Failed to load the model. Please check the model path.")
            raise

        self.model.eval()

        #Define transform for the model 
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])

        #set threshold
        self.threshold = threshold
        self.iou_threshold = iou_threshold
        self.confidence_threshold = confidence_threshold

        #Performance tracking 
        self.frame_times = []
        self.detection_count = 0


    def to_tensor_from_red_channel(self, image_bgr):
        red_channel = image_bgr[:, :, 2].astype(np.float32)/255
        return torch.from_numpy(red_channel).unsqueeze(0).unsqueeze(0).contiguous()


    def rotate_tensor_image(self, tensor_img,angle_deg):
        angle_rad = math.radians(angle_deg)
        theta = torch.tensor([
            [math.cos(angle_rad), -math.sin(angle_rad), 0],
            [math.sin(angle_rad), math.cos(angle_rad), 0]
        ], dtype=torch.float32,device = tensor_img.device )
        grid = F.affine_grid(theta.unsqueeze(0), tensor_img.size(), align_corners=False)
        rotated_tensor = F.grid_sample(tensor_img, grid,padding_mode='zeros', align_corners=False)
        return rotated_tensor
    
    def get_rotated_templates(self,template_tensor,angles):
        rotated_templates =[]
        for angle in angles:
            rotated_template = self.rotate_tensor_image(template_tensor, angle)
            rotated_template -= rotated_template.mean()
            rotated_templates.append(rotated_template)
        return torch.cat(rotated_templates, dim=0)
    
    def match_template_batched(self,image_tensor, templates_btch):
        N,_, H, W = templates_btch.shape
        response = F.conv2d(image_tensor,templates_btch,stride=1)
        response_np = response.squeeze(0).detach().cpu().numpy()

        all_boxes = []
        for i in range(N):
            r = response_np[i]
            yx = np.where(r>self.threshold)
            for y, x in zip(*yx):
                score = float(r[y, x])
                all_boxes.append((x, y, x + W, y + H, score))
        return all_boxes
    

    def nms_torch(self, boxes):
        if not boxes:
            return []

        boxes = np.array(boxes)
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        scores = boxes[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]  # sort by score (descending)

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(tuple(boxes[i]))

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter)

            order = order[1:][iou <= self.iou_threshold]

        return keep


    def extract_crops(self, frame, boxes,padding = 5):
        crops =[]
        scaled_boxes = []

        frame_h,frame_w = frame.shape[:2]
        for x1,y1,x2,y2,_ in boxes:
            # added padding befor cropping for better regions
            x1_pad = max(0, int(x1 - padding))
            y1_pad = max(0, int(y1 - padding))  
            x2_pad = min(frame_w, int(x2 + padding))
            y2_pad = min(frame_h, int(y2 + padding))

            #extract crop 
            crop = frame[y1_pad:y2_pad, x1_pad:x2_pad]
            if crop.size >0:
                crops.append(crop)
                scaled_boxes.append((x1_pad, y1_pad, x2_pad, y2_pad))
        
        return crops, scaled_boxes
    
    def process_with_cnn(self, crops,scaled_boxes):
        if not crops:
            return []
        
        batch_tensors = []
        for crop in crops:
            rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_crop = Image.fromarray(rgb_crop)
            tensor = self.transform(pil_crop)
            batch_tensors.append(tensor)

        batch = torch.stack(batch_tensors).to(self.device)

        with torch.no_grad():
            class_outputs,centroid_outputs = self.model(batch)

        detections = []
        for i, ((x1, y1, x2, y2), class_output, centroid_output) in enumerate(
                zip(scaled_boxes, class_outputs, centroid_outputs)):
            
            probability = class_output.item()
            
            # If confidence is high enough
            if probability > self.confidence_threshold:
                # Convert normalized centroid to crop coordinates
                crop_w, crop_h = x2 - x1, y2 - y1
                crop_cx = int(centroid_output[0].item() * crop_w)
                crop_cy = int(centroid_output[1].item() * crop_h)
                
                # Convert to frame coordinates
                frame_cx = x1 + crop_cx
                frame_cy = y1 + crop_cy
                
                detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'centroid': (frame_cx, frame_cy),
                    'probability': probability
                })
        
        return detections
    
    def process_frame(self,frame):
        #resize the frame to 200x200 for better performance

        original_h,original_w = frame.shape[:2]
        process_size = (150, 150)
        resized_frame = cv2.resize(frame, process_size)

        #strat time
        start_time = time.time()
        
        # Convert the resized frame to tensor
        image_tensor = self.to_tensor_from_red_channel(resized_frame).to(self.device)

        # get regions
        all_boxes = self.match_template_batched(image_tensor,self.rotated_templates)
        filtered_boxes = self.nms_torch(all_boxes)

        # Extract crops from the original frame
        crops,scaled_boxes = self.extract_crops(resized_frame, filtered_boxes)

        #Process crops with CNN
        detections = self.process_with_cnn(crops,scaled_boxes)

        #convert detections to original frame size
        scale_x = original_w / process_size[0]
        scale_y = original_h / process_size[1]
        
        for detection in detections:
                x1, y1, x2, y2 = detection['bbox']
                cx,cy = detection['centroid']

                detection['bbox']=(
                    int(x1 * scale_x),
                    int(y1 * scale_y),
                    int(x2 * scale_x),
                    int(y2 * scale_y)

                )

                detection['centroid'] = (
                    int(cx * scale_x),
                    int(cy * scale_y)
                )


        #Precess time
        process_time = time.time() - start_time
        self.frame_times.append(process_time)
        if len(self.frame_times) > 30:
            self.frame_times.pop(0)
        
        fps = 1.0 / (sum(self.frame_times) / len(self.frame_times))
        self.detection_count += len(detections)
        
        return {
            'detections': detections,
            'process_time': process_time,
            'fps': fps
        }
    
    def draw_results(self, frame, results):
        """Draw detection results on the frame"""
        output = frame.copy()
        
        
        for det in results['detections']:
            # Draw bounding box
            x1, y1, x2, y2 = det['bbox']
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw centroid
            cx, cy = det['centroid']
            cv2.circle(output, (cx, cy), 5, (0, 0, 255), -1)
            cv2.circle(output, (cx, cy), 8, (0, 0, 255), 2)
            
            # Show confidence
            prob = det['probability']
            cv2.putText(output, f"{prob:.2f}", (x1, y1-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw FPS
        cv2.putText(output, f"FPS: {results['fps']:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 25), 2)
        cv2.putText(output, f"Time: {results['process_time']*1000:.1f}ms", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 25, 255), 2)
        cv2.putText(output, f"Total Detections: {self.detection_count}", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 2, 255), 2)
        
        return output
    

    
def run_real_time_detection(template_path, model_path, camera_index=0):
    """Run real-time detection with the combined pipeline"""
    # Initialize detector
    detector = CombinedDetector(
        template_path=template_path,
        model_path=model_path,
        threshold=80,        # Template matching threshold
        iou_threshold=0.01,     # NMS threshold
        confidence_threshold=0.01  # CNN confidence threshold
    )
    
    # Open camera
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_index}")
        return
    
    print("Press 'q' to quit, 's' to save screenshot")
    
    # Main loop
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break
        
        # Resize to target size (640x640)
        frame = cv2.resize(frame, (540, 540))
        
        # Process frame
        results = detector.process_frame(frame)
        
        # Draw results
        output = detector.draw_results(frame, results)
        
        # Show frame
        cv2.imshow("L-Bend Detection", output)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            cv2.imwrite(f"detection_{timestamp}.png", output)
            print(f"Screenshot saved: detection_{timestamp}.png")
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    template_path = "image_20250321_201450_lbent_1.png"  # Your template image
    model_path = "lbend_detector_scripted.pt"  # Your trained CNN model
    camera_index =1 # Change if needed
    
    with torch.no_grad():
        run_real_time_detection(template_path, model_path, camera_index)


       
         
    
