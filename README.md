# 🚗 Smart Traffic Analyzer 🚦

![Project Banner](https://via.placeholder.com/800x200/2D3748/FFFFFF?text=Vehicle_Detection) <!-- Replace with actual banner -->

A real-time vehicle detection, tracking, and analytics system powered by YOLOv8 and DeepSORT for intelligent traffic monitoring.

## 🌟 Features

<div align="center">
  <img src="[https://via.placeholder.com/400x225/4A5568/FFFFFF?text=Detection+Demo](https://i.ibb.co/0yrh7fBS/Screenshot-2.png)" alt="Detection Demo" width="45%">
  <img src="[https://via.placeholder.com/400x225/4A5568/FFFFFF?text=Counting+Demo](https://i.ibb.co/rRWjFTmV/Screenshot-2025-07-07-163403.png)" alt="Counting Demo" width="45%">
</div>

- **Accurate Vehicle Detection** (Cars, Trucks, Buses, Motorcycles)
- **Dual-Lane Traffic Counting** with directional analysis
- **Speed Estimation** in km/h
- **Real-time Analytics Dashboard**
- **Comprehensive Data Logging** (JSON output)
- **Customizable Detection Zones**
- **Cross-platform Compatibility**

## 🛠️ Technologies Used

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.7-orange?logo=opencv)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-red)
![DeepSORT](https://img.shields.io/badge/DeepSORT-RealTime-green)
![NumPy](https://img.shields.io/badge/NumPy-1.24+-yellow?logo=numpy)

## 📦 Installation

### Prerequisites
- Python 3.8+
- NVIDIA GPU (Recommended for best performance)
- FFmpeg (For video processing)

### Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/smart-traffic-analyzer.git
cd smart-traffic-analyzer

# Install dependencies
pip install -r requirements.txt

# Download YOLOv8 weights (automatically done on first run)

🚀 Usage
Basic Command : python imgcounting.py --input sample.mp4 --output results.avi

### Configuration Options

Parameter	Default	Description
--input	None	Input video file path
--output	None	Output video file path
--line_y	240	Y-position of counting line
--confidence	0.5	Detection confidence threshold
--classes	2,3,5,7	Vehicle classes to detect
--show	False	Show real-time display

🎥 Demonstration
<div align="center"> <img src="https://via.placeholder.com/600x338/4A5568/FFFFFF?text=Live+Detection+GIF" alt="Live Detection"> <p><em>Real-time vehicle detection and counting demonstration</em></p> </div>
🤝 Contributing
Fork the project

Create your feature branch (git checkout -b feature/AmazingFeature)

Commit your changes (git commit -m 'Add some amazing feature')

Push to the branch (git push origin feature/AmazingFeature)

Open a Pull Request

📜 License
Distributed under the MIT License. See LICENSE for more information.

📧 Contact
Jeet Bhatt - echolima1323@gmail.com
Project Link: https://github.com/bhattjee/VEHICLE-DETECTION
