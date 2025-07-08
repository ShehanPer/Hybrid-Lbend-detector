import cv2
import numpy as np
import torch
import torch.nn.functional as F
import time
import math

class CombinedDetector:
    def __init__(self, template_path, model_path, threshold=0.52, iou_threshold=0.3, confidence_threshold=0.5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        template_bgr = cv2.imread(template_path)
        template_bgr = cv2.resize(template_bgr, (64, 64))
        self.template_tensor = self.to_tensor_from_red_channel(template_bgr).to(self.device)

        angles = [0, 45, 90, 135, 180, 225, 270, 315]
        self.rotated_templates = self.get_rotated_templates(self.template_tensor, angles)

        try:
            self.model = torch.jit.load(model_path, map_location=self.device)
            print("Model loaded successfully.")
        except:
            print("Failed to load model.")
            raise

        self.model.eval()

        

        self.threshold = threshold
        self.iou_threshold = iou_threshold
        self.confidence_threshold = confidence_threshold
        self.frame_times = []
        self.detection_count = 0

    def preprocess(self,image_rgb):
        #image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(image_rgb, (64, 64)).astype(np.float32) / 255.0
        tensor = torch.from_numpy(resized).permute(2, 0, 1)
        return tensor
    
    def to_tensor_from_red_channel(self, image_bgr):
        red_channel = image_bgr[:, :, 2].astype(np.float32) / 255
        return torch.from_numpy(red_channel).unsqueeze(0).unsqueeze(0).contiguous()

    def rotate_image_with_padding(self, image_np, angle):
        h, w = image_np.shape
        center = (w // 2, h // 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos, sin = np.abs(rot_mat[0, 0]), np.abs(rot_mat[0, 1])
        nW, nH = int(h * sin + w * cos), int(h * cos + w * sin)
        rot_mat[0, 2] += (nW / 2) - center[0]
        rot_mat[1, 2] += (nH / 2) - center[1]
        return cv2.warpAffine(image_np, rot_mat, (nW, nH), borderValue=255)

    def get_rotated_templates(self, template_tensor, angles):
        rotated_np_templates = []
        for angle in angles:
            template_np = template_tensor.squeeze().cpu().numpy()
            rotated_np = self.rotate_image_with_padding(template_np, angle)
            rotated_np_templates.append(rotated_np)
        self.rotated_np_templates = rotated_np_templates
        return rotated_np_templates

    def match_template_opencv(self, image_np):
        all_boxes = []
        h, w = 64, 64
        image_gray = (image_np[:, :, 2] / 255.0).astype(np.float32)
        for template_np in self.rotated_np_templates:
            result = cv2.matchTemplate(image_gray, template_np, cv2.TM_CCOEFF_NORMED)
            yx = np.where(result >= self.threshold)
            for y, x in zip(*yx):
                score = result[y, x]
                all_boxes.append((x, y, x + w, y + h, score))
        return all_boxes

    def nms_numpy(self, boxes):
        if not boxes:
            return []
        boxes = np.array(boxes)
        x1, y1, x2, y2, scores = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3], boxes[:, 4]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []

        while order.size:
            i = order[0]
            keep.append(tuple(boxes[i]))
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            order = order[1:][iou <= self.iou_threshold]
        return keep

    def extract_crops(self, frame, boxes, padding=5):
        crops = []
        scaled_boxes = []
        frame_h, frame_w = frame.shape[:2]
        for x1, y1, x2, y2, _ in boxes:
            x1_pad = max(0, int(x1 - padding))
            y1_pad = max(0, int(y1 - padding))
            x2_pad = min(frame_w, int(x2 + padding))
            y2_pad = min(frame_h, int(y2 + padding))
            crop = frame[y1_pad:y2_pad, x1_pad:x2_pad]
            if crop.size > 0:
                crops.append(crop)
                scaled_boxes.append((x1_pad, y1_pad, x2_pad, y2_pad))
        return crops, scaled_boxes

    def process_with_cnn(self, crops, scaled_boxes):
        if not crops:
            return []
        batch_tensors = []
        for crop in crops:
            rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensor = self.preprocess(rgb_crop)
            batch_tensors.append(tensor)

        batch = torch.stack(batch_tensors).to(self.device)
        with torch.no_grad():
            class_outputs, centroid_outputs = self.model(batch)
        detections = []
        for i, ((x1, y1, x2, y2), class_output, centroid_output) in enumerate(zip(scaled_boxes, class_outputs, centroid_outputs)):
            probability = class_output.item()
            if probability > self.confidence_threshold:
                crop_w, crop_h = x2 - x1, y2 - y1
                crop_cx = int(centroid_output[0].item() * crop_w)
                crop_cy = int(centroid_output[1].item() * crop_h)
                frame_cx = x1 + crop_cx
                frame_cy = y1 + crop_cy
                detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'centroid': (frame_cx, frame_cy),
                    'probability': probability
                })
        return detections

    def process_frame(self, frame):
        original_h, original_w = frame.shape[:2]
        process_size = (150, 150)
        resized_frame = cv2.resize(frame, process_size)
        start_time = time.time()
        all_boxes = self.match_template_opencv(resized_frame)
        filtered_boxes = self.nms_numpy(all_boxes)
        crops, scaled_boxes = self.extract_crops(resized_frame, filtered_boxes)
        detections = self.process_with_cnn(crops, scaled_boxes)
        scale_x = original_w / process_size[0]
        scale_y = original_h / process_size[1]
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            cx, cy = detection['centroid']
            detection['bbox'] = (int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y))
            detection['centroid'] = (int(cx * scale_x), int(cy * scale_y))
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
        output = frame.copy()
        for det in results['detections']:
            x1, y1, x2, y2 = det['bbox']
            cx, cy = det['centroid']
            prob = det['probability']
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(output, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(output, f"{prob:.2f}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(output, f"FPS: {results['fps']:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 25), 2)
        cv2.putText(output, f"Time: {results['process_time']*1000:.1f}ms", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 25, 255), 2)
        cv2.putText(output, f"Total Detections: {self.detection_count}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 2, 255), 2)
        return output

def run_real_time_detection(template_path, model_path, camera_index=0):
    detector = CombinedDetector(template_path, model_path)
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_index}")
        return
    print("Press 'q' to quit")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break

        frame = cv2.resize(frame, (540, 540))
        results = detector.process_frame(frame)
        output = detector.draw_results(frame, results)
        detector.detection_count = 0
        cv2.imshow("L-Bend Detection", output)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_real_time_detection("image_20250321_201450_lbent_1.png", "lbend_detector_scripted.pt", camera_index=1)
