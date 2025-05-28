import cv2
import numpy as np
import matplotlib.pyplot as plt

def rotate_image_with_padding(image, angle):
    """Rotate image with proper padding to avoid cropping"""
    (h, w) = image.shape[:2]
    center = (w//2, h//2)
    
    # Calculate rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Calculate new bounding dimensions
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    
    # Adjust rotation matrix for new center
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    
    # Perform rotation with new dimensions
    rotated = cv2.warpAffine(image, M, (new_w, new_h), 
                           flags=cv2.INTER_LINEAR, 
                           borderMode=cv2.BORDER_CONSTANT, 
                           borderValue=0)
    return rotated

def frequency_domain_template_match(full_img, template):
    """Perform template matching in frequency domain"""
    H, W = full_img.shape
    h, w = template.shape
    
    # Pad dimensions for proper convolution
    pad_H = H + h - 1
    pad_W = W + w - 1
    
    # FFT of full image
    F = np.fft.fft2(full_img, s=(pad_H, pad_W))
    
    # FFT of template (flipped for correlation)
    template_flipped = np.flip(template)
    T = np.fft.fft2(template_flipped, s=(pad_H, pad_W))
    # Apply high-pass filter to both F and T in frequency domain
    def high_pass_filter(shape, radius=10):
        rows, cols = shape
        crow, ccol = rows // 2, cols // 2
        mask = np.ones((rows, cols), np.float32)
        y, x = np.ogrid[:rows, :cols]
        center_dist = np.sqrt((x - ccol)**2 + (y - crow)**2)
        mask[center_dist < radius] = 0
        return mask

    hpf_mask = high_pass_filter((pad_H, pad_W), radius=10)
    F = F * hpf_mask
    T = T * hpf_mask
    # Cross-correlation in frequency domain
    R = F * np.conj(T)
    corr = np.fft.ifft2(R)
    corr = np.abs(corr)
    
    # Extract valid correlation region
    valid_corr = corr[:H-h+1, :W-w+1]
    
    # Normalize correlation
    if valid_corr.max() > valid_corr.min():
        valid_corr_norm = (valid_corr - valid_corr.min()) / (valid_corr.max() - valid_corr.min())
    else:
        valid_corr_norm = valid_corr
    
    return valid_corr_norm

def get_bounding_boxes_from_heatmap(heatmap, template_shape, threshold=0.5, min_area=50):
    """Extract bounding boxes from correlation heatmap"""
    h_tmpl, w_tmpl = template_shape
    
    # Threshold heatmap
    _, thresh = cv2.threshold((heatmap*255).astype(np.uint8), 
                             int(threshold*255), 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    for cnt in contours:
        if cv2.contourArea(cnt) > min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            # Adjust box size to match template dimensions
            boxes.append((x, y, max(w, w_tmpl), max(h, h_tmpl)))
    
    return boxes

def non_max_suppression(boxes, scores, overlap_thresh=0.3):
    """Apply non-maximum suppression to remove overlapping boxes"""
    if len(boxes) == 0:
        return []
    
    # Convert to numpy array
    boxes = np.array(boxes, dtype=np.float32)
    
    # Calculate areas
    areas = boxes[:, 2] * boxes[:, 3]
    
    # Sort by scores in descending order
    indices = np.argsort(scores)[::-1]
    
    keep = []
    while len(indices) > 0:
        # Pick the box with highest score
        i = indices[0]
        keep.append(i)
        
        if len(indices) == 1:
            break
            
        # Calculate IoU with remaining boxes
        xx1 = np.maximum(boxes[i, 0], boxes[indices[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[indices[1:], 1])
        xx2 = np.minimum(boxes[i, 0] + boxes[i, 2], boxes[indices[1:], 0] + boxes[indices[1:], 2])
        yy2 = np.minimum(boxes[i, 1] + boxes[i, 3], boxes[indices[1:], 1] + boxes[indices[1:], 3])
        
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        intersection = w * h
        
        union = areas[i] + areas[indices[1:]] - intersection
        iou = intersection / union
        
        # Keep boxes with low IoU
        indices = indices[1:][iou <= overlap_thresh]
    
    return [boxes[i] for i in keep]

def main():
    # Load template image (grayscale)
    cropped_img = cv2.imread('image_20250321_201450_lbent_1.png', cv2.IMREAD_GRAYSCALE)
    if cropped_img is None:
        print("Error: Could not load template image")
        return
    
    # Resize template to reasonable size (increased from 48)
    max_dim = 120  # Increased template size
    h, w = cropped_img.shape
    scale = min(max_dim / h, max_dim / w) if max(h, w) > max_dim else 1.0
    
    new_h, new_w = int(h * scale), int(w * scale)
    resized_template = cv2.resize(cropped_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Create multiple scales of the template
    scales = [ 1.0]  # Multiple scales for better detection
    
    # Create rotated versions at different scales
    angles = np.arange(0, 360, 90)  # Every thita degrees
    templates = []
    
    for scale in scales:
        scaled_h, scaled_w = int(new_h * scale), int(new_w * scale)
        if scaled_h > 0 and scaled_w > 0:
            scaled_template = cv2.resize(resized_template, (scaled_w, scaled_h), 
                                       interpolation=cv2.INTER_AREA)
            
            for angle in angles:
                rotated = rotate_image_with_padding(scaled_template, angle)
                templates.append(rotated)
    
    # Load full image (grayscale)
    full_img = cv2.imread('Dataset\\test\\images\\image_20250321_200726_lbent_4.png', cv2.IMREAD_GRAYSCALE)
    if full_img is None:
        print("Error: Could not load full image")
        return
    
    print(f"Processing {len(templates)} template variations...")
    
    # Compute correlation heatmaps for all template variations
    all_heatmaps = []
    all_scores = []
    all_template_shapes = []
    
    for i, template in enumerate(templates):
        if i % 10 == 0:
            print(f"Processing template {i+1}/{len(templates)}")
            
        heatmap = frequency_domain_template_match(full_img, template)
        all_heatmaps.append(heatmap)
        all_scores.append(heatmap.max())
        all_template_shapes.append(template.shape)
    
    # Combine heatmaps (take maximum response)
    combined_heatmap = np.zeros_like(all_heatmaps[0])
    for heatmap in all_heatmaps:
        # Resize heatmaps to same size if needed
        if heatmap.shape != combined_heatmap.shape:
            heatmap_resized = cv2.resize(heatmap, 
                                       (combined_heatmap.shape[1], combined_heatmap.shape[0]))
        else:
            heatmap_resized = heatmap
        combined_heatmap = np.maximum(combined_heatmap, heatmap_resized)
    
    # Adaptive threshold
    mean_val = combined_heatmap.mean()
    std_val = combined_heatmap.std()
    threshold = mean_val + 1.5 * std_val  # Reduced multiplier
    threshold = np.clip(threshold, 0.2, 0.5)  # Keep within bounds
    
    print(f"Using threshold: {threshold:.3f}")
    
    # Get bounding boxes
    template_shape = resized_template.shape
    boxes = get_bounding_boxes_from_heatmap(combined_heatmap, template_shape, 
                                          threshold=threshold, min_area=200)
    
    # Apply non-maximum suppression
    if boxes:
        scores = [combined_heatmap[int(y):int(y+h), int(x):int(x+w)].max() 
                 for x, y, w, h in boxes]
        boxes = non_max_suppression(boxes, scores, overlap_thresh=0.3)
    
    print(f"Found {len(boxes)} detections")
    
    # Draw results
    img_color = cv2.cvtColor(full_img, cv2.COLOR_GRAY2BGR)
    for i, (x, y, w, h) in enumerate(boxes):
        cv2.rectangle(img_color, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
        cv2.putText(img_color, f'{i+1}', (int(x), int(y-5)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Display results
    plt.figure(figsize=(16, 8))
    
    plt.subplot(2, 3, 1)
    plt.title("Original Template")
    plt.imshow(cropped_img, cmap='gray')
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.title("Resized Template")
    plt.imshow(resized_template, cmap='gray')
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.title("Full Image")
    plt.imshow(full_img, cmap='gray')
    plt.axis('off')
    
    plt.subplot(2, 3, 4)
    plt.title("Correlation Heatmap")
    plt.imshow(combined_heatmap, cmap='hot')
    plt.colorbar()
    plt.axis('off')
    
    plt.subplot(2, 3, 5)
    plt.title("Thresholded Heatmap")
    thresh_display = (combined_heatmap > threshold).astype(float)
    plt.imshow(thresh_display, cmap='gray')
    plt.axis('off')
    
    plt.subplot(2, 3, 6)
    plt.title(f"Detection Result ({len(boxes)} found)")
    plt.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()