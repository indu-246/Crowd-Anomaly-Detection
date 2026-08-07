# Crowd Anomaly Detection

## Overview

Crowd Anomaly Detection is a computer vision-based system developed to identify unusual activities in crowded environments. The project analyzes crowd movement from surveillance videos and detects behaviors that significantly deviate from normal motion patterns. The objective is to support automated monitoring in public spaces and improve situational awareness in real-time.

---

## Objectives

- Detect abnormal crowd behavior from surveillance footage.
- Learn normal crowd movement patterns.
- Identify deviations using anomaly scoring.
- Improve crowd safety through automated monitoring.

---

## Methodology

The proposed system follows the following workflow:

1. Video frames are captured and preprocessed.
2. Frames are resized and normalized for uniform processing.
3. Optical Flow is computed between consecutive frames to capture motion information.
4. Motion features are extracted and represented using Mixture of Dynamic Textures (MDT).
5. The trained model analyzes the extracted features.
6. An anomaly score is generated for each observation.
7. Based on the anomaly score, the scene is classified as **Normal** or **Abnormal**.

---

## System Workflow

```
Input Video
      │
      ▼
Frame Extraction
      │
      ▼
Preprocessing
      │
      ▼
Optical Flow Feature Extraction
      │
      ▼
Mixture of Dynamic Textures (MDT)
      │
      ▼
Anomaly Score Computation
      │
      ▼
Normal / Abnormal Prediction
```

---

## Technologies Used

- Python
- OpenCV
- NumPy
- Scikit-Image
- TensorFlow
- Joblib

---

## Key Features

- Automated crowd behavior analysis
- Optical Flow-based motion feature extraction
- MDT-based anomaly detection
- Pre-trained machine learning model
- Frame-level anomaly prediction
- Supports surveillance video analysis

---

## Applications

- Smart City Surveillance
- Railway and Metro Stations
- Airports
- Shopping Malls
- Stadiums
- Public Events
- Traffic Monitoring
- Crowd Safety Management

---

## Future Scope

The current implementation focuses on anomaly detection using motion analysis. Future enhancements include:

- Direct MP4 video processing
- Real-time CCTV monitoring
- YOLOv8-based person detection
- Multi-object tracking using ByteTrack
- Automatic alert generation
- Suspicious region localization
- Saving abnormal frames with timestamps

---

## Conclusion

This project demonstrates an effective approach for detecting abnormal crowd behavior by learning normal motion patterns and identifying deviations through anomaly scoring. The proposed system can assist surveillance applications by providing automated and reliable monitoring of crowded environments while reducing manual observation efforts.
