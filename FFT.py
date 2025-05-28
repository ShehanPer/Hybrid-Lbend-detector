import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

def rotate_image_with_padding(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Calculate bounding box of the rotated image
    cos = np.abs(rot_mat[0, 0])
    sin = np.abs(rot_mat[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # Adjust rotation matrix to account for translation
    rot_mat[0, 2] += (new_w / 2) - center[0]
    rot_mat[1, 2] += (new_h / 2) - center[1]

    # Perform actual rotation
    rotated = cv2.warpAffine(image, rot_mat, (new_w, new_h), flags=cv2.INTER_LINEAR, borderValue=0)
    return rotated

def match_grid_correlation(full_img, templates, cell_size, threshold=0.45):
    h, w = full_img.shape
    ch, cw = cell_size
    bboxes = []

    for y in range(0, h - ch + 1, ch):
        for x in range(0, w - cw + 1, cw):
            patch = full_img[y:y+ch, x:x+cw]
            if patch.shape != (ch, cw):
                continue

            max_corr = 0
            for temp in templates:
                temp = cv2.resize(temp, (cw, ch), interpolation=cv2.INTER_AREA)
                result = cv2.matchTemplate(patch, temp, cv2.TM_CCOEFF_NORMED)
                corr_val = result[0][0]
                max_corr = max(max_corr, corr_val)

            if max_corr > threshold:
                bboxes.append(((x, y), (x + cw, y + ch)))

    return bboxes

def draw_bboxes(img, bboxes):
    out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for pt1, pt2 in bboxes:
        cv2.rectangle(out, pt1, pt2, (0, 255, 0), 2)
    return out

def main():
    start = time.time()
    # Load template image
    cropped_img = cv2.imread('image_20250321_201450_lbent_1.png', cv2.IMREAD_GRAYSCALE)
    if cropped_img is None:
        print("Error: Could not load template image")
        return
    
    # Resize template to a manageable fixed size
    max_dim = 613
    h, w = cropped_img.shape
    scale = min(max_dim / h, max_dim / w) if max(h, w) > max_dim else 1.0
    new_h, new_w = int(h * scale), int(w * scale)
    resized_template = cv2.resize(cropped_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Generate rotated templates
    angles = np.arange(0, 360, 15)
    templates = [rotate_image_with_padding(resized_template, angle) for angle in angles]

    # Load full image
    full_img = cv2.imread('Dataset/test/images/image_20250321_200726_lbent_4.png', cv2.IMREAD_GRAYSCALE)
    if full_img is None:
        print("Error: Could not load full image")
        return

    # Perform grid-based correlation matching
    cell_size = (new_h, new_w)
    bboxes = match_grid_correlation(full_img, templates, cell_size, threshold=0.3)

    # Draw and show
    result = draw_bboxes(full_img, bboxes)
    end = time.time()
    print(f"Processing time: {end - start:.2f} seconds")
    plt.figure(figsize=(12, 8))
    plt.imshow(result)
    plt.title("Detected Regions")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()
