# Pattern-Based Object Detection Notebook Guide

This guide provides detailed information about the `rpn_fft.ipynb` notebook in the Pattern-Based Object Detection project.

## Overview

The notebook demonstrates different implementations of template-based object detection algorithms, from a basic FFT-based approach to more advanced PyTorch-accelerated methods. Each section builds on the previous one, showing improvements in performance, accuracy, and processing speed.

## Notebook Structure

The notebook is organized into four main sections:

### 1. FFT-Based Implementation

This section implements object detection using Fast Fourier Transform techniques:

```python
# Key functions:
def high_pass_filter_fft(image):
    # Applies a high-pass filter using FFT to remove low-frequency components
    # and highlight edges in the image

def rotate_image_with_padding(image, angle):
    # Rotates an image with proper padding to avoid cropping edges

def convolve_and_get_bboxes(image, template, threshold):
    # Performs convolution between image and template, returning bounding boxes
    # above the specified threshold

def non_max_suppression_fast(boxes, iou_thresh=0.3):
    # Removes overlapping detections, keeping the ones with highest scores

def average_boxes(boxes, iou_thresh=0.5):
    # Averages similar boxes to produce more stable detections
```

The implementation demonstrates:
- How to process images in the frequency domain using FFT
- How to handle rotation invariance with template variations
- How to apply non-maximum suppression to refine detections
- How to visualize and score object detection results

### 2. OpenCV Template Matching

This section uses OpenCV's built-in template matching methods:

```python
def match_template(image, template, threshold=0.5):
    # Uses cv2.matchTemplate to find instances of the template in the image
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    yx = np.where(result >= threshold)
    h, w = template.shape
    return [(x, y, x + w, y + h, result[y, x]) for y, x in zip(*yx)]
```

The implementation demonstrates:
- How to use OpenCV's efficient template matching algorithms
- How to compare with the custom FFT approach
- How to extract bounding boxes from the correlation map
- Performance benchmarking against the FFT approach

### 3. PyTorch Accelerated Implementation

This section leverages PyTorch for GPU acceleration:

```python
def match_template_batched(image_tensor, templates_batch, threshold=0.48):
    # Uses PyTorch's Conv2d to perform template matching in a batched manner
    # for multiple templates at once
    N, _, h, w = templates_batch.shape
    response = F.conv2d(image_tensor, templates_batch, stride=1)
    response_np = response.squeeze(0).detach().cpu().numpy()
    
    # Extract bounding boxes from the response map
    all_boxes = []
    for i in range(N):
        r = response_np[i]
        yx = np.where(r >= threshold)
        for y, x in zip(*yx):
            score = r[y, x]
            all_boxes.append((x, y, x + w, y + h, score))
    return all_boxes
```

The implementation demonstrates:
- How to convert OpenCV images to PyTorch tensors
- How to batch process multiple rotated templates at once
- How to leverage GPU acceleration for faster processing
- How to use PyTorch's grid sampling for image rotation
- Integration with torchvision's NMS implementation
- Significant performance improvements over CPU-based methods

### 4. CNN Model Training for Verification and Centroid Prediction

This section implements a Convolutional Neural Network (CNN) model that serves two purposes:
- Verifies if the template matches are true positives
- Predicts the exact centroid coordinates within the detected regions

```python
class CNNDetector(nn.Module):
    def __init__(self):
        super(CNNDetector, self).__init__()
        # Shared feature extraction
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Classification branch
        self.classifier = nn.Sequential(
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Centroid prediction branch
        self.regressor = nn.Sequential(
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Sigmoid()  # Normalized coordinates (0-1)
        )
```

The implementation demonstrates:
- How to create a multi-task CNN model for both classification and regression
- How to prepare training data from CSV annotations
- How to train the model with a combined loss function
- How to export the model to TorchScript for deployment
- How to visualize training metrics and model performance

Training results are visualized with plots showing:
- Classification accuracy (reaching >95%)
- Centroid prediction error (measured in pixels)
- Combined loss curves for training and validation sets
- Example predictions on test images

The trained model is exported to TorchScript format (`lbend_detector_scripted.pt`) for efficient deployment on edge devices.

### 5. Real-time Inference

This section builds on the PyTorch implementation for real-time video processing:

```python
# Main processing loop for real-time detection
while True:
    ret, frame = cap.read()
    # Process the frame with the PyTorch model
    # Display detection results with FPS count
```

The implementation demonstrates:
- How to integrate with video streams or IP cameras
- How to optimize for real-time performance
- How to calculate and display frames per second (FPS)
- How to visualize detections in real-time

## Edge-Optimized Implementation

The notebook techniques are implemented in the standalone Python script `L_bent_detector_without_transformers.py`, which:
- Provides a lightweight, transformer-free implementation optimized for edge devices
- Achieves 20 FPS performance on Jetson Nano hardware
- Uses a hybrid approach combining template matching for region proposals and CNN for verification
- Features batch processing of candidate regions for efficient inference
- Includes real-time visualization with FPS metrics and detection count

This edge-optimized version makes several performance improvements:
- Processes smaller input resolutions (150x150) for template matching
- Uses OpenCV's efficient matchTemplate implementation
- Applies Non-Maximum Suppression to filter duplicate detections
- Utilizes the lightweight CNN for verification and precise centroid localization
- Prioritizes inference speed while maintaining detection accuracy

## Results and Visualization

Each section includes visualization of the detection results:
- Bounding boxes around detected objects
- Confidence scores for each detection
- Performance metrics (processing time, FPS)
- Comparison between different approaches

The notebook provides a comprehensive comparison of different object detection techniques, showing how to progress from basic implementations to optimized, real-time capable solutions suitable for edge deployment.
