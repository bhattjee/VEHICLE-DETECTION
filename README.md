# Vehicle Detection & Face Recognition System

A comprehensive computer vision system featuring real-time vehicle detection, tracking, and analytics alongside advanced face recognition capabilities. Built with state-of-the-art deep learning models including YOLOv8, DeepSORT, and custom feature extraction algorithms.

## Features

### Vehicle Detection & Analytics
- **Accurate Multi-Class Detection** - Cars, trucks, buses, motorcycles, and trains
- **Dual-Lane Traffic Counting** - Separate counting for left and right lanes with directional analysis
- **Real-Time Speed Estimation** - Calculate vehicle speeds in km/h with smoothing algorithms
- **Advanced Object Tracking** - DeepSORT integration for robust vehicle tracking across frames
- **Interactive Analytics Dashboard** - Real-time visualization with FPS, progress, and statistics
- **Comprehensive Data Logging** - JSON export with detailed vehicle information and timestamps
- **Customizable Detection Zones** - Configurable counting lines and regions of interest
- **Performance Monitoring** - Frame-by-frame detection statistics and speed metrics

### Face Recognition System
- **Advanced Feature Extraction** - LBP, HOG, Gabor filters, statistical, and geometric features
- **Quality-Controlled Sample Collection** - Real-time quality assessment during face enrollment
- **Multi-Metric Similarity Calculation** - Cosine similarity, Euclidean distance, and Manhattan distance
- **Prediction Stability** - Temporal smoothing for consistent recognition results
- **Adjustable Recognition Threshold** - Tunable sensitivity for different use cases
- **Persistent Training Data** - Save and load face encodings with automatic scaling
- **Real-Time Recognition** - Live camera feed with confidence scores

## Technologies Used

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8-orange?logo=opencv)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-red)
![DeepSORT](https://img.shields.io/badge/DeepSORT-RealTime-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1-orange?logo=pytorch)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange?logo=scikit-learn)
![NumPy](https://img.shields.io/badge/NumPy-1.24-yellow?logo=numpy)

## Project Structure

```
VEHICLE-DETECTION/
├── face_detection/
│   ├── facedetection.py              # Basic face detection using Haar cascades
│   ├── facedetectionwithname.py      # Advanced face recognition system
│   └── haarcascade_*.xml             # Pre-trained cascade classifiers
├── image_counting/
│   ├── imgcounting.py                # Vehicle detection and counting system
│   └── yolov8*.pt                    # YOLO model weights
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

## Installation

### Prerequisites
- Python 3.8 or higher
- NVIDIA GPU (recommended for optimal performance)
- FFmpeg (for video processing)

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/bhattjee/VEHICLE-DETECTION.git
   cd VEHICLE-DETECTION
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download YOLOv8 weights**
   - The weights will be automatically downloaded on first run
   - Alternatively, download manually from [Ultralytics](https://github.com/ultralytics/ultralytics)

## Usage

### Vehicle Detection & Counting

Navigate to the image counting directory:
```bash
cd image_counting
```

Run the vehicle detection system:
```bash
python imgcounting.py
```

**Note:** Update the `VIDEO_FILE` path in `imgcounting.py` to point to your video file.

#### Interactive Controls
- **`q`** - Quit the application
- **`s`** - Save current data to JSON
- **`r`** - Restart video from beginning
- **`SPACE`** - Pause/Resume playback
- **`d`** - Toggle debug mode

#### Configuration Parameters
Edit the constants at the top of `imgcounting.py`:
- `FRAME_WIDTH` / `FRAME_HEIGHT` - Output resolution
- `COUNTING_LINE_Y` - Position of the counting line (0-1 ratio of frame height)
- `CONFIDENCE_THRESHOLD` - Detection confidence (0.0-1.0)
- `VEHICLE_CLASSES` - COCO class IDs to detect
- `PIXEL_TO_METER_RATIO` - Calibration for speed estimation

### Face Recognition

Navigate to the face detection directory:
```bash
cd face_detection
```

Run the advanced face recognition system:
```bash
python facedetectionwithname.py
```

#### Menu Options
1. **Add new face** - Collect training samples for a person
2. **Start face recognition** - Real-time recognition from camera
3. **List known faces** - View all enrolled faces with statistics
4. **Delete face** - Remove a person from the system
5. **Adjust recognition threshold** - Tune sensitivity (0.1-0.95)
6. **System info** - View configuration details
7. **Exit** - Quit the application

#### Basic Face Detection
For simple face detection without recognition:
```bash
python facedetection.py
```

## Output Data

### Vehicle Detection
The system generates JSON files containing:
- Vehicle counts per lane
- Speed measurements (km/h)
- Timestamps and frame numbers
- Bounding box coordinates
- Detection statistics

Example output filename: `vehicle_data_YYYYMMDD_HHMMSS.json`

### Face Recognition
Training data is persisted in:
- `face_encodings.pkl` - Feature vectors and scaler
- `face_names.json` - Name mappings and counters
- `face_training_data/person_*/` - Sample images

## Demonstration

<div align="center">
  <img src="https://i.ibb.co/rRWjFTmV/Screenshot-2025-07-07-163403.png" alt="Vehicle Detection" width="45%">
  <img src="https://i.ibb.co/0yrh7fBS/Screenshot-2.png" alt="Face Recognition" width="45%">
</div>

## Troubleshooting

### Common Issues

**YOLO model not loading**
- Ensure internet connection for automatic download
- Manually download `yolov8s.pt` from Ultralytics releases

**Camera not opening**
- Check camera permissions
- Verify camera index (change `cv2.VideoCapture(0)` to appropriate index)

**Low detection accuracy**
- Adjust `CONFIDENCE_THRESHOLD` in configuration
- Ensure adequate lighting conditions
- For face recognition, collect more training samples

**Performance issues**
- Use smaller YOLO model (`yolov8n.pt` instead of `yolov8s.pt`)
- Reduce frame resolution in configuration
- Enable GPU acceleration with CUDA-compatible PyTorch

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

**Jeet Bhatt** - jeetbhatt1323@gmail.com

**Project Link:** https://github.com/bhattjee/VEHICLE-DETECTION/

## Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv8
- [DeepSORT](https://github.com/ZQPei/deep_sort_pytorch) for tracking algorithms
- [OpenCV](https://opencv.org/) for computer vision utilities
- COCO Dataset for pre-trained models
