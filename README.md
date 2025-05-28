# Pattern-Based Object Detection

A computer vision project that implements pattern/template-based object detection using Fast Fourier Transform (FFT) and frequency domain techniques.

## Overview

This project provides advanced algorithms for detecting objects in images based on template matching in the frequency domain. The implementation offers several advantages over traditional spatial domain template matching:

- **Rotation invariance**: Detects objects at different orientations
- **Multiple scale detection**: Identifies objects at various scales
- **Frequency domain filtering**: Uses high-pass filters to enhance edge detection
- **Optimized for speed**: Leverages FFT for efficient correlation computation

## Features

- **FFT-Based Template Matching**: Performs correlation in the frequency domain for faster computation
- **High-Pass Filtering**: Highlights edges and fine details while suppressing low-frequency components
- **Rotation with Padding**: Handles template rotation without cropping edges
- **Non-Maximum Suppression**: Intelligently merges overlapping detections
- **Grid-based Correlation**: Divides the search space into cells for efficient processing
- **Region Proposal Network (RPN) Style Approach**: Combines FFT with RPN-like techniques

## Project Structure

- `connected_com.py` - Main implementation of frequency domain template matching
- `FFT.py` - Core FFT utilities and grid-based correlation matching
- `rpn_fft.ipynb` - Jupyter notebook demonstrating RPN-style approach with FFT
- `Results/` - Directory containing example detection outputs and visualizations

## Notebook Content

The `rpn_fft.ipynb` notebook provides a comprehensive demonstration of the project's object detection capabilities with multiple implementation approaches:

1. **FFT-Based Implementation**:
   - High-pass filtering in the frequency domain
   - Template rotation with padding
   - Convolution-based template matching
   - Non-maximum suppression for merging detections
   - Average boxes algorithm for clustering similar detections

2. **OpenCV Template Matching**:
   - Traditional template matching using OpenCV's matchTemplate
   - Multiple rotation angles for rotation invariance
   - Score thresholding and bounding box extraction
   - Comparison with FFT-based approach

3. **PyTorch Accelerated Implementation**:
   - GPU-accelerated template matching using PyTorch's Conv2d
   - Batched processing of multiple rotated templates
   - Tensor-based image rotation
   - Integration with torchvision's NMS implementation
   - Performance benchmarking against other methods

4. **Real-time Inference**:
   - Video stream processing for real-time detection
   - FPS measurement and optimization
   - Integration with IP camera streams
   
Each section includes visualization of the detection results and performance metrics, making it easy to compare the different approaches.

## Usage

```python
# Example usage
import cv2
from connected_com import frequency_domain_template_match, get_bounding_boxes_from_heatmap

# Load images
template = cv2.imread('template.png', cv2.IMREAD_GRAYSCALE)
image = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE)

# Perform template matching in frequency domain
correlation_map = frequency_domain_template_match(image, template)

# Get bounding boxes from the correlation map
threshold = 0.5
boxes = get_bounding_boxes_from_heatmap(correlation_map, template.shape, threshold)

# Draw boxes on the original image
# ...
```

## Algorithm Details

1. **Preprocessing**: 
   - Convert images to grayscale
   - Apply high-pass filtering in frequency domain to enhance edges

2. **Template Preparation**:
   - Generate rotated versions of the template
   - Create multiple scales of the template if needed

3. **FFT-Based Matching**:
   - Compute FFT of both image and template
   - Perform convolution in frequency domain (multiplication)
   - Convert back to spatial domain using inverse FFT

4. **Post-processing**:
   - Threshold correlation map
   - Extract connected components as candidate regions
   - Apply non-maximum suppression to remove overlapping detections

## Results

The `Results/` directory contains example outputs showing:
- Detection results with bounding boxes
- Correlation heatmaps
- FFT analysis visualizations
- Filtering effects
- Comparison between different implementation approaches

## Dependencies

- Python 3.x
- OpenCV (cv2)
- NumPy
- Matplotlib
- Jupyter Notebook (for running the `.ipynb` files)
- PyTorch
- torchvision

## License

This project is licensed under the MIT License - see the LICENSE file for details.
