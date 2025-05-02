# 👤 Professional Face Recognition System

![App Screenshot](screenshot.png) 

A comprehensive face recognition solution with real-time detection, user management, and configurable settings.

## ✨ Features

| Feature | Status |
|---------|--------|
| ✅ Real-time face detection and recognition | Implemented |
| ⚙️ Configurable settings (tolerance, themes) | Implemented |
| 📝 Face registration and management | Planned |
| 📊 Recognition history logging | Implemented |
| 📁 Import/export functionality | Implemented |
| 🎥 Webcam integration with FPS monitoring | Implemented |

## 📂 File Structure

```
face-recognition-app/
│
├── dataset/                  # User data storage
│   ├── config.pkl            # Application configuration
│   ├── face_recognition_v2.db # SQLite database
│   └── known_faces_v2.pkl    # Face encodings
│
├── src/                      # Source code
│   └── app.py                # Main application
│
├── requirements.txt          # Dependencies
├── README.md                 # Documentation
└── LICENSE                   # MIT Licens

```

## 🚀 Installation

**Prerequisites**

Python 3.8+
Webcam


# 1. Clone repository

git clone https://github.com/Ahmed230460/Face-Recognition-App.git

cd Face-Recognition-App

# 2. Install dependencies

pip install -r requirements.txt

# 3. Install required packages

pip install opencv-python face-recognition ttkbootstrap pillow numpy


## 💻 Usage

python src/app.py


## Key Functions

User Management

Register new faces

Edit/delete existing users

Face Operations

Real-time detection

Recognition system

System Controls

Adjust recognition tolerance

Change UI themes

View system logs

## ⚙️Configuration

**config.pkl**:  Application settings	

**face_recognition_v2.db**:  User database

**known_faces_v2.pkl**:  Face encodings


## Run the application:

python app.py


## Use the UI to:

Start/Stop face capture and recognition

Add/edit/delete users with images

View recognition history and logs

Adjust settings via the menu (theme, tolerance, sound, etc.)



## Configuration

Settings are stored in config.pkl

Auto-saves every 5 minutes

Editable via the Settings menu

## License

© 2025 Ahmed Dawood. All Rights Reserved. Distributed under the MIT License.

## Documentation

For more details, visit: (https://github.com/Ahmed230460/Face-Recognition-App)



