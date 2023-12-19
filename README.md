# CircaTrack
A Customized Circle Detection and Tracking System in Videos with Speed and Acceleration Prediction and Re-tracking mechanism
# CircaTrack
This is my very first computer vision project that I completed within three days. Thank you for taking the time to check it out. Your support and feedback mean a lot! If this project proves helpful to you, don't hesitate to give it a star!

## Introduction

This repository includes three main Python files:

### HughCircleHyperTuner.py

This file fine-tunes parameters for HoughCircles to detect circles accurately within a video.

### CircaTrack.py

CircaTrack utilizes a customized tracking system to track circles in a video. It draws bounding boxes around detected circles and numbers them in the center. To use:
python CircaTrack.py --video [video_path] --output_video [video_path]

Instructions:
- Replace [video_path] with the path to your input and output video file.

#### Features:
- **Speed and Acceleration Prediction:** Utilizes the last n frames to predict the next positions, incorporating both speed and acceleration.
- **Re-tracking System:** Enhances robustness by re-assigning unselected circles based on similarities and thresholds.

### CSRT_KCF_Track.py
This file allows the user to choose between CSRT and KCF trackers to track circles within a video. It provides outputs similar to CircaTrack. To use:
python CSRT_KCF_Track.py --input_video [video_path] --output_video [video_path] --tracker [csrt/kcf]

Instructions:
- Replace [video_path] with the path to your input and output video file.
- Choose either csrt or kcf as the tracker type.

## Input and Outputs
Input: Video file

Outputs:
- Video with boxed circles and numbered centers
- TXT file recording positions for each circle

## Superiority
The customized tracking system implemented in CircaTrack showcases superior performance compared to both CSRT and KCF trackers.
