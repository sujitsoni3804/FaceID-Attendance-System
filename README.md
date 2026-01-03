# 🎯 FaceID Attendance System

A real-time face recognition-based attendance system built with Python. Automatically detects and identifies faces to log check-in/check-out times.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📝 **Face Registration** | Register new users with multi-frame face encoding |
| 🔍 **Real-time Recognition** | Instant face detection and identification via webcam |
| ⏰ **Check-in/Check-out** | Track attendance with timestamps and validation |
| 🎯 **Face Alignment** | Ensures frontal face detection for accuracy |
| 📊 **CSV Logging** | Automatic attendance records with date/time stamps |
| 🖥️ **Full-screen UI** | Clean interface with visual feedback and animations |

---

## 🛠️ Tech Stack

- **Language:** Python 3.8+
- **Face Detection:** MediaPipe Face Mesh
- **Face Recognition:** face_recognition (dlib)
- **Computer Vision:** OpenCV
- **Data Storage:** CSV, Pickle

---

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- Webcam
- Windows/Linux/Mac

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/FaceID-Attendance-System.git
cd FaceID-Attendance-System
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** dlib installation may require CMake and Visual Studio Build Tools on Windows.

### Step 4: Run the Application

```bash
python app.py
```

---

## 🎮 Usage

### Controls

| Key | Action |
|-----|--------|
| `R` | Registration Mode |
| `A` | Attendance Mode |
| `1` | Switch to Check-In |
| `2` | Switch to Check-Out |
| `Q` | Quit |

### Registration Mode

1. Enter the user's name when prompted
2. Face the camera directly until 7 frames are captured
3. Encoding is saved automatically

### Attendance Mode

1. Select Check-In (`1`) or Check-Out (`2`)
2. Face the camera – recognition happens automatically
3. Attendance is logged to `attendance_log.csv`

---

## 📁 Project Structure

```
FaceID-Attendance-System/
├── app.py                 # Main application
├── requirements.txt       # Python dependencies
├── attendance_log.csv     # Attendance records
├── db/                    # Stored face encodings (.pickle)
└── Detection_images/      # Captured detection snapshots
```

---

## 🙏 Acknowledgments

- [MediaPipe](https://github.com/google/mediapipe) for face mesh detection
- [face_recognition](https://github.com/ageitgey/face_recognition) for face encoding
- [OpenCV](https://opencv.org/) for computer vision

---

<p align="center">
  Made with ❤️ for Smart Attendance Management
</p>
