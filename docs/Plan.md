# Measure It - Project Plan

## Overview
Develop an AI-based object measurement system
that measures the width and height of a book
using a mobile camera.

## Technologies
- Python
- OpenCV
- PyTorch
- Mask R-CNN
- CVAT
- Git and GitHub

## Implementation Steps

1. Camera calibration using OpenCV.
2. Capture and label 70+ book images.
3. Split dataset into training, validation
   and testing.
4. Train a Mask R-CNN segmentation model.
5. Build image-to-millimetres measurement.
6. Evaluate accuracy using a calliper.
7. Compare raw and undistorted measurements.

## Selected Approach
Use OpenCV for camera calibration and
PyTorch Mask R-CNN for segmentation.

## Rejected Approach
Ultralytics YOLO and Roboflow models
are excluded because the assessment
does not allow them.

## Limitations
Advanced testing and optimization may
be limited by the eight-hour deadline.
Unfinished work will be documented honestly.