
# Professional Face Recognition System

![App Screenshot](screenshot.png) <!-- Add a screenshot later -->

A robust face recognition application with user management, real-time detection, and recognition capabilities.

## Features

- 👤 Face registration and management
- 🔍 Real-time face detection and recognition
- 📊 Recognition history logging
- ⚙️ Configurable settings (tolerance, themes, etc.)
- 📁 Import/export functionality
- 📷 Webcam integration with FPS monitoring

## File Structure
face-recognition-app/
│
│ ├── config.pkl # Application configuration
│ ├── face_recognition_v2.db # Main SQLite database
│ └── known_faces_v2.pkl # Face encodings data
│
├── src/ # Source code
│ ├── app.py # Main application file
│
├── requirements.txt # Python dependencies
├── README.md # This documentation
└── LICENSE # Project license


## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Ahmed230460/Face-Recognition-App.git
   cd Face-Recognition-App

2.**Install dependencies**:
pip install -r requirements.txt

3.**Required Python Packages**:
pip install opencv-python face-recognition tk ttkbootstrap pillow numpy sqlite3

##Usage

1.**Run the application**:

python src/app.py

2.**Main Functions**:

1-Add new users with face capture

2-Start/stop face recognition

3-View recognition history

4-Manage user database

3.**Configuration**:

The application automatically creates these configuration files:

config.pkl: Stores application settings (theme, tolerance, etc.)
face_recognition_v2.db:	SQLite database for user data
known_faces_v2.pkl:	Serialized face encodings

##License

Distributed under the MIT License.

##Contact

Ahmed Dawood - ahmeddawood0001@gmail.com

Project Link: https://github.com/Ahmed230460/Face-Recognition-App
   
