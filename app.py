import cv2
import numpy as np
import sqlite3
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, filedialog
from PIL import Image, ImageTk, ImageOps
import os
import datetime
import face_recognition
import pickle
import csv
from ttkbootstrap import Style
import threading
import winsound
import time
from collections import deque
import webbrowser

class ProfessionalFaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Professional Face Recognition System")
        self.root.geometry("1200x850")
        self.root.minsize(1000, 750)
        
        # Performance monitoring
        self.frame_count = 0
        self.fps = 0
        self.last_fps_update = time.time()
        self.processing_times = deque(maxlen=20)
            # Add these attributes
        self.is_capturing = False
        self.recognizing = False
        self.tolerance = 0.6
        # Enhanced face tracking
        self.known_face_encodings = []
        self.known_face_ids = []
        self.known_face_names = []
        self.face_tracking = {}  # For tracking face movement
        self.current_recognitions = {}  # For displaying recognition info
        
        # Setup enhanced database
        self.db_conn = self.setup_database()
        
        # Load configuration
        self.load_config()
        
        # Initialize UI
        self.setup_ui()
        
        # Initialize camera with error handling
        self.camera_init()
        
        # Start background tasks
        self.start_background_tasks()
        
        # Register cleanup
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def setup_database(self):
        """Enhanced database setup with error handling"""
        try:
            conn = sqlite3.connect('face_recognition_v2.db', check_same_thread=False)
            cursor = conn.cursor()
            
            # Enhanced user table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    role TEXT,
                    department TEXT,
                    last_seen TIMESTAMP,
                    first_registered TIMESTAMP,
                    image_path TEXT,
                    notes TEXT
                )
            ''')
            
            # Recognition history table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS recognition_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    recognition_time TIMESTAMP,
                    confidence REAL,
                    FOREIGN KEY(user_id) REFERENCES users(id)
                )
            ''')
            
            conn.commit()
            return conn
        except sqlite3.Error as e:
            messagebox.showerror("Database Error", f"Failed to initialize database: {str(e)}")
            raise

    def load_config(self):
        """Load application configuration"""
        self.config = {
            'tolerance': 0.55,
            'min_face_size': 100,
            'recognition_interval': 5,
            'enable_sound': True,
            'enable_logging': True,
            'auto_save_interval': 300,
            'theme': 'darkly'
        }
        
        # Try to load from config file
        try:
            if os.path.exists('config.pkl'):
                with open('config.pkl', 'rb') as f:
                    saved_config = pickle.load(f)
                    self.config.update(saved_config)
        except Exception as e:
            self.log_error(f"Failed to load config: {str(e)}")

    def save_config(self):
        """Save current configuration"""
        try:
            with open('config.pkl', 'wb') as f:
                pickle.dump(self.config, f)
        except Exception as e:
            self.log_error(f"Failed to save config: {str(e)}")

    def camera_init(self):
        """Initialize camera with enhanced error handling"""
        self.cap = None
        self.camera_index = 0
        self.camera_resolution = (1280, 720)
        
        # Try multiple camera indexes
        for i in range(3):
            try:
                self.cap = cv2.VideoCapture(i)
                if self.cap.isOpened():
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_resolution[0])
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_resolution[1])
                    self.log(f"Camera initialized at index {i}")
                    break
            except Exception as e:
                self.log_error(f"Camera init error at index {i}: {str(e)}")
        
        if not self.cap or not self.cap.isOpened():
            self.show_camera_error()

    def show_camera_error(self):
        """Show camera error with recovery options"""
        error_window = tk.Toplevel(self.root)
        error_window.title("Camera Error")
        error_window.geometry("400x200")
        
        tk.Label(error_window, text="Camera Initialization Failed", font=('Arial', 14)).pack(pady=10)
        tk.Label(error_window, text="Please check your camera connection").pack()
        
        btn_frame = tk.Frame(error_window)
        btn_frame.pack(pady=20)
        
        tk.Button(btn_frame, text="Retry", command=lambda: [self.camera_init(), error_window.destroy()]).pack(side=tk.LEFT, padx=10)
        tk.Button(btn_frame, text="Continue Without Camera", command=error_window.destroy).pack(side=tk.LEFT, padx=10)
        tk.Button(btn_frame, text="Exit", command=self.root.quit).pack(side=tk.LEFT, padx=10)

    def setup_ui(self):
        """Enhanced UI setup with modern features"""
        # Apply theme
        self.style = Style(theme=self.config['theme'])
        
        # Main container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Camera and controls
        left_panel = ttk.Frame(main_container)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Camera frame
        self.camera_frame = ttk.LabelFrame(left_panel, text="Live Camera Feed", padding=5)
        self.camera_frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(self.camera_frame, bg='black')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Info overlay
        self.info_label = ttk.Label(self.camera_frame, text="FPS: 0 | Faces: 0", style='info.Inverse.TLabel')
        self.info_label.place(relx=0.01, rely=0.01, anchor=tk.NW)
        
        # Control buttons
        control_frame = ttk.Frame(left_panel)
        control_frame.pack(fill=tk.X, pady=5)
        
        self.btn_capture = ttk.Button(control_frame, text="Start Capture", command=self.toggle_capture, style='primary.TButton')
        self.btn_capture.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        
        self.btn_recognize = ttk.Button(control_frame, text="Start Recognition", command=self.toggle_recognition, style='success.TButton')
        self.btn_recognize.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        
        self.btn_add_user = ttk.Button(control_frame, text="Add User", command=self.show_add_user_dialog, style='info.TButton')
        self.btn_add_user.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        
        # Right panel - User management and logs
        right_panel = ttk.Frame(main_container, width=300)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y)
        
        # User management
        user_mgmt_frame = ttk.LabelFrame(right_panel, text="User Management", padding=5)
        user_mgmt_frame.pack(fill=tk.X, pady=5)
        
        self.user_tree = ttk.Treeview(user_mgmt_frame, columns=("ID", "Name", "Last Seen"), show="headings", height=10)
        self.user_tree.heading("ID", text="ID")
        self.user_tree.heading("Name", text="Name")
        self.user_tree.heading("Last Seen", text="Last Seen")
        self.user_tree.column("ID", width=50, anchor=tk.CENTER)
        self.user_tree.column("Name", width=120)
        self.user_tree.column("Last Seen", width=120)
        self.user_tree.pack(fill=tk.BOTH, expand=True)
        
        # User action buttons
        user_btn_frame = ttk.Frame(user_mgmt_frame)
        user_btn_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(user_btn_frame, text="Edit", command=self.edit_user, style='warning.TButton').pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        ttk.Button(user_btn_frame, text="Delete", command=self.delete_user, style='danger.TButton').pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        ttk.Button(user_btn_frame, text="Refresh", command=self.load_users, style='secondary.TButton').pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        
        # Recognition history
        history_frame = ttk.LabelFrame(right_panel, text="Recent Recognitions", padding=5)
        history_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.history_tree = ttk.Treeview(history_frame, columns=("Time", "Name", "Confidence"), show="headings", height=8)
        self.history_tree.heading("Time", text="Time")
        self.history_tree.heading("Name", text="Name")
        self.history_tree.heading("Confidence", text="Confidence")
        self.history_tree.column("Time", width=120)
        self.history_tree.column("Name", width=100)
        self.history_tree.column("Confidence", width=80)
        self.history_tree.pack(fill=tk.BOTH, expand=True)
        
        # Log console
        log_frame = ttk.LabelFrame(right_panel, text="System Log", padding=5)
        log_frame.pack(fill=tk.BOTH, pady=5)
        
        self.log_text = tk.Text(log_frame, height=8, bg="#1e1e1e", fg="white", wrap=tk.WORD, state=tk.DISABLED)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status_bar = ttk.Label(self.root, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        
        # Menu bar
        self.setup_menu()
        
        # Load initial data
        self.load_users()
        self.load_known_faces()
        self.load_recognition_history()

    def setup_menu(self):
        """Setup menu bar with enhanced options"""
        menubar = tk.Menu(self.root)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Export Users...", command=self.export_users)
        file_menu.add_command(label="Import Users...", command=self.import_users)
        file_menu.add_separator()
        file_menu.add_command(label="Settings...", command=self.show_settings)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_close)
        menubar.add_cascade(label="File", menu=file_menu)
        
        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        view_menu.add_command(label="Full Screen", command=self.toggle_fullscreen, accelerator="F11")
        view_menu.add_separator()
        
        # Theme submenu
        theme_menu = tk.Menu(view_menu, tearoff=0)
        for theme in ['darkly', 'superhero', 'flatly', 'cosmo', 'litera']:
            theme_menu.add_command(label=theme.capitalize(), 
                                 command=lambda t=theme: self.change_theme(t))
        view_menu.add_cascade(label="Theme", menu=theme_menu)
        menubar.add_cascade(label="View", menu=view_menu)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="Documentation", command=self.show_docs)
        help_menu.add_command(label="About", command=self.show_about)
        menubar.add_cascade(label="Help", menu=help_menu)
        
        self.root.config(menu=menubar)
        
        # Bind keyboard shortcuts
        self.root.bind("<F11>", lambda e: self.toggle_fullscreen())
        self.root.bind("<F1>", lambda e: self.show_docs())

    def toggle_fullscreen(self):
        """Toggle fullscreen mode"""
        self.root.attributes("-fullscreen", not self.root.attributes("-fullscreen"))

    def change_theme(self, theme):
        """Change application theme"""
        self.style.theme_use(theme)
        self.config['theme'] = theme
        self.save_config()
        self.log(f"Theme changed to {theme}")

    def show_settings(self):
        """Show settings dialog"""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Settings")
        settings_window.geometry("500x400")
        
        # Tolerance setting
        ttk.Label(settings_window, text="Recognition Tolerance:").pack(pady=(10, 0))
        tolerance_scale = ttk.Scale(settings_window, from_=0.4, to=0.8, value=self.config['tolerance'])
        tolerance_scale.pack(fill=tk.X, padx=20, pady=5)
        
        # Other settings
        sound_var = tk.BooleanVar(value=self.config['enable_sound'])
        ttk.Checkbutton(settings_window, text="Enable Sounds", variable=sound_var).pack(anchor=tk.W, padx=20, pady=5)
        
        log_var = tk.BooleanVar(value=self.config['enable_logging'])
        ttk.Checkbutton(settings_window, text="Enable Logging", variable=log_var).pack(anchor=tk.W, padx=20, pady=5)
        
        # Save button
        ttk.Button(settings_window, text="Save Settings", 
                  command=lambda: self.save_settings(
                      tolerance_scale.get(),
                      sound_var.get(),
                      log_var.get()
                  )).pack(pady=20)

    def save_settings(self, tolerance, enable_sound, enable_logging):
        """Save application settings"""
        self.config.update({
            'tolerance': float(tolerance),
            'enable_sound': enable_sound,
            'enable_logging': enable_logging
        })
        self.save_config()
        self.log("Settings saved")
        messagebox.showinfo("Settings", "Settings saved successfully")

    def show_add_user_dialog(self):
        """Enhanced add user dialog with more fields"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Add New User")
        dialog.geometry("400x400")
        
        ttk.Label(dialog, text="Full Name:").pack(pady=(10, 0))
        name_entry = ttk.Entry(dialog)
        name_entry.pack(fill=tk.X, padx=20, pady=5)
        
        ttk.Label(dialog, text="Role:").pack(pady=(5, 0))
        role_entry = ttk.Entry(dialog)
        role_entry.pack(fill=tk.X, padx=20, pady=5)
        
        ttk.Label(dialog, text="Department:").pack(pady=(5, 0))
        dept_entry = ttk.Entry(dialog)
        dept_entry.pack(fill=tk.X, padx=20, pady=5)
        
        ttk.Label(dialog, text="Notes:").pack(pady=(5, 0))
        notes_text = tk.Text(dialog, height=4)
        notes_text.pack(fill=tk.X, padx=20, pady=5)
        
        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(btn_frame, text="Cancel", command=dialog.destroy).pack(side=tk.RIGHT, padx=5)
        ttk.Button(btn_frame, text="Save", 
                  command=lambda: self.save_new_user(
                      name_entry.get(),
                      role_entry.get(),
                      dept_entry.get(),
                      notes_text.get("1.0", tk.END),
                      dialog
                  )).pack(side=tk.RIGHT, padx=5)

    def save_new_user(self, name, role, department, notes, dialog):
        """Save new user with enhanced information"""
        if not name.strip():
            messagebox.showerror("Error", "Name cannot be empty")
            return
            
        try:
            # Capture face image
            ret, frame = self.cap.read()
            if not ret:
                messagebox.showerror("Error", "Failed to capture image")
                return
                
            # Detect and encode face
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_encodings = face_recognition.face_encodings(rgb_frame)
            
            if not face_encodings:
                messagebox.showerror("Error", "No face detected in the image")
                return
                
            # Save to database
            cursor = self.db_conn.cursor()
            timestamp = datetime.datetime.now()
            image_path = f'dataset/User_{timestamp.strftime("%Y%m%d_%H%M%S")}.jpg'
            
            cursor.execute('''
                INSERT INTO users 
                (name, role, department, last_seen, first_registered, image_path, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (name, role, department, timestamp, timestamp, image_path, notes))
            
            user_id = cursor.lastrowid
            self.db_conn.commit()
            
            # Save face encoding and image
            self.known_face_encodings.append(face_encodings[0])
            self.known_face_ids.append(user_id)
            self.known_face_names.append(name)
            self.save_known_faces()
            
            if not os.path.exists('dataset'):
                os.makedirs('dataset')
            cv2.imwrite(image_path, frame)
            
            # Update UI
            self.load_users()
            self.log(f"User {name} added with ID {user_id}")
            self.status("User added successfully")
            
            if self.config['enable_sound']:
                winsound.PlaySound("SystemAsterisk", winsound.SND_ASYNC)
                
            dialog.destroy()
            
        except Exception as e:
            self.log_error(f"Failed to add user: {str(e)}")
            messagebox.showerror("Error", f"Failed to add user: {str(e)}")

    def load_users(self):
        """Load users from database with error handling"""
        try:
            # Clear current tree
            for item in self.user_tree.get_children():
                self.user_tree.delete(item)
                
            cursor = self.db_conn.cursor()
            cursor.execute('SELECT id, name, last_seen FROM users ORDER BY name')
            
            for user in cursor.fetchall():
                self.user_tree.insert("", tk.END, values=user)
                
        except Exception as e:
            self.log_error(f"Failed to load users: {str(e)}")

    def load_known_faces(self):
        """Load known faces with error handling"""
        try:
            if os.path.exists('known_faces_v2.pkl'):
                with open('known_faces_v2.pkl', 'rb') as f:
                    data = pickle.load(f)
                    self.known_face_encodings = data['encodings']
                    self.known_face_ids = data['ids']
                    self.known_face_names = data['names']
                    
            if not self.known_face_encodings:
                self.log("No known faces loaded")
                
        except Exception as e:
            self.log_error(f"Failed to load known faces: {str(e)}")

    def save_known_faces(self):
        """Save known faces with error handling"""
        try:
            with open('known_faces_v2.pkl', 'wb') as f:
                pickle.dump({
                    'encodings': self.known_face_encodings,
                    'ids': self.known_face_ids,
                    'names': self.known_face_names
                }, f)
        except Exception as e:
            self.log_error(f"Failed to save known faces: {str(e)}")

    def load_recognition_history(self, limit=20):
        """Load recognition history"""
        try:
            for item in self.history_tree.get_children():
                self.history_tree.delete(item)
                
            cursor = self.db_conn.cursor()
            cursor.execute('''
                SELECT h.recognition_time, u.name, h.confidence 
                FROM recognition_history h
                JOIN users u ON h.user_id = u.id
                ORDER BY h.recognition_time DESC
                LIMIT ?
            ''', (limit,))
            
            for record in cursor.fetchall():
                time_str = record[0].strftime("%Y-%m-%d %H:%M:%S") if isinstance(record[0], datetime.datetime) else record[0]
                confidence = f"{float(record[2])*100:.1f}%" if record[2] else "N/A"
                self.history_tree.insert("", tk.END, values=(time_str, record[1], confidence))
                
        except Exception as e:
            self.log_error(f"Failed to load history: {str(e)}")

    def toggle_capture(self):
        """Toggle face capture mode"""
        self.is_capturing = not self.is_capturing
        self.btn_capture.config(text="Stop Capture" if self.is_capturing else "Start Capture")
        self.status("Capture " + ("started" if self.is_capturing else "stopped"))
        self.log(f"Face capture {'enabled' if self.is_capturing else 'disabled'}")

    def toggle_recognition(self):
        """Toggle face recognition mode"""
        if not self.known_face_encodings:
            messagebox.showwarning("Warning", "No users registered. Please add users first.")
            return
            
        self.recognizing = not self.recognizing
        self.btn_recognize.config(text="Stop Recognition" if self.recognizing else "Start Recognition")
        
        if self.recognizing:
            self.recognition_thread = threading.Thread(target=self.recognize_faces, daemon=True)
            self.recognition_thread.start()
            self.status("Recognition started")
            self.log("Face recognition enabled")
        else:
            self.status("Recognition stopped")
            self.log("Face recognition disabled")

    def recognize_faces(self):
        """Enhanced face recognition with tracking and confidence"""
        last_recognition_time = 0  # This should be at method start
        
        while self.recognizing and self.cap and self.cap.isOpened():
            start_time = time.time()
            
            try:
                ret, frame = self.cap.read()
                if not ret:
                    self.log_error("Failed to read frame")
                    time.sleep(0.1)
                    continue
                    
                # Resize frame for faster processing
                small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                
                # Detect faces
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                
                # Update face tracking
                self.update_face_tracking(face_locations)
                
                # Process each face
                current_time = time.time()
                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                    # Scale back up face locations
                    top *= 2; right *= 2; bottom *= 2; left *= 2
                    
                    # Only recognize periodically to reduce CPU load
                    if current_time - last_recognition_time > self.config['recognition_interval']:
                        matches = face_recognition.compare_faces(
                            self.known_face_encodings, 
                            face_encoding,
                            tolerance=self.config['tolerance']
                        )
                        
                        if True in matches:
                            first_match_index = matches.index(True)
                            user_id = self.known_face_ids[first_match_index]
                            name = self.known_face_names[first_match_index]
                            
                            # Calculate confidence
                            face_distances = face_recognition.face_distance(
                                self.known_face_encodings, 
                                face_encoding
                            )
                            confidence = 1 - face_distances[first_match_index]
                            
                            # Update database
                            self.update_recognition_history(user_id, confidence)
                            
                            # Update UI in main thread
                            self.root.after(0, lambda: self.update_recognition_ui(
                                user_id, name, confidence, (left, top, right, bottom)))
                            
                            last_recognition_time = current_time  # This should be inside the if block
            
            except Exception as e:
                self.log_error(f"Recognition error: {str(e)}")
                time.sleep(1)
                
        self.log("Recognition stopped")

    def update_face_tracking(self, face_locations):
        """Track face movement between frames"""
        # This is a simplified tracking implementation
        # In a production app, you'd use a more sophisticated tracker
        current_ids = list(self.face_tracking.keys())
        
        for i, loc in enumerate(face_locations):
            matched = False
            for face_id in current_ids:
                # Simple distance-based matching
                prev_loc = self.face_tracking[face_id]['location']
                distance = ((loc[0]-prev_loc[0])**2 + (loc[1]-prev_loc[1])**2)**0.5
                
                if distance < 50:  # Threshold for same face
                    self.face_tracking[face_id] = {
                        'location': loc,
                        'last_seen': time.time()
                    }
                    matched = True
                    break
                    
            if not matched:
                new_id = max(self.face_tracking.keys()) + 1 if self.face_tracking else 1
                self.face_tracking[new_id] = {
                    'location': loc,
                    'last_seen': time.time()
                }
                
        # Remove faces not seen recently
        current_time = time.time()
        for face_id in list(self.face_tracking.keys()):
            if current_time - self.face_tracking[face_id]['last_seen'] > 2:  # 2 seconds timeout
                del self.face_tracking[face_id]

    def update_recognition_history(self, user_id, confidence):
        """Record recognition in database"""
        try:
            cursor = self.db_conn.cursor()
            timestamp = datetime.datetime.now()
            
            # Update user's last seen time
            cursor.execute('''
                UPDATE users 
                SET last_seen = ?
                WHERE id = ?
            ''', (timestamp, user_id))
            
            # Add to recognition history
            cursor.execute('''
                INSERT INTO recognition_history 
                (user_id, recognition_time, confidence)
                VALUES (?, ?, ?)
            ''', (user_id, timestamp, float(confidence)))
            
            self.db_conn.commit()
            
        except Exception as e:
            self.log_error(f"Failed to update recognition history: {str(e)}")

    def update_recognition_ui(self, user_id, name, confidence, location):
        """Update UI with recognition results"""
        try:
            # Update history tree
            time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            confidence_str = f"{confidence*100:.1f}%"
            self.history_tree.insert("", 0, values=(time_str, name, confidence_str))
            
            # Keep only last 20 items
            if len(self.history_tree.get_children()) > 20:
                self.history_tree.delete(self.history_tree.get_children()[-1])
                
            # Update current recognitions
            self.current_recognitions[user_id] = {
                'name': name,
                'confidence': confidence,
                'location': location,
                'time': time.time()
            }
            
            # Play sound if enabled
            if self.config['enable_sound']:
                winsound.PlaySound("SystemExclamation", winsound.SND_ASYNC)
                
            self.log(f"Recognized {name} (ID: {user_id}) with {confidence_str} confidence")
            
        except Exception as e:
            self.log_error(f"Failed to update recognition UI: {str(e)}")

    def update_performance_metrics(self, start_time, face_count):
        """Update FPS and performance metrics"""
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        
        self.frame_count += 1
        if time.time() - self.last_fps_update >= 1:
            self.fps = self.frame_count / (time.time() - self.last_fps_update)
            self.frame_count = 0
            self.last_fps_update = time.time()
            
            avg_process_time = sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0
            self.info_label.config(text=f"FPS: {self.fps:.1f} | Faces: {face_count} | Avg: {avg_process_time*1000:.1f}ms")

    def update_frame(self):
        """Update camera frame with annotations"""
        if self.cap and self.cap.isOpened():
            try:
                ret, frame = self.cap.read()
                
                if ret:
                    # Only do face detection if capturing or recognizing
                    if hasattr(self, 'is_capturing') and self.is_capturing:
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        face_locations = face_recognition.face_locations(rgb_frame)
                        
                        for (top, right, bottom, left) in face_locations:
                            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    
                    # Convert to PhotoImage and update display
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame)
                    img_tk = ImageTk.PhotoImage(img)
                    
                    self.canvas.img_tk = img_tk  # Keep reference
                    self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
                    
            except Exception as e:
                self.log_error(f"Frame update error: {str(e)}")
                
        self.root.after(10, self.update_frame)

    def draw_face_annotations(self, frame):
        """Draw face detection and recognition annotations"""
        # Draw detection boxes
        if self.is_capturing:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            
            for (top, right, bottom, left) in face_locations:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        
        # Draw recognition info
        current_time = time.time()
        for user_id, data in list(self.current_recognitions.items()):
            if current_time - data['time'] > 5:  # Show recognition for 5 seconds
                del self.current_recognitions[user_id]
                continue
                
            left, top, right, bottom = data['location']
            confidence = data['confidence']
            name = data['name']
            
            # Draw rectangle
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            
            # Draw label background
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            
            # Draw text
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"{confidence*100:.1f}%", (left + 6, bottom - 20), font, 0.5, (255, 255, 255), 1)

    def edit_user(self):
        """Edit user information"""
        selected = self.user_tree.selection()
        if not selected:
            messagebox.showwarning("Warning", "Please select a user to edit")
            return
            
        user_id = self.user_tree.item(selected)['values'][0]
        
        # Fetch current user data
        cursor = self.db_conn.cursor()
        cursor.execute('SELECT name, role, department, notes FROM users WHERE id = ?', (user_id,))
        user_data = cursor.fetchone()
        
        if not user_data:
            messagebox.showerror("Error", "Selected user not found")
            return
            
        # Create edit dialog
        dialog = tk.Toplevel(self.root)
        dialog.title(f"Edit User ID {user_id}")
        dialog.geometry("400x400")
        
        ttk.Label(dialog, text="Full Name:").pack(pady=(10, 0))
        name_entry = ttk.Entry(dialog)
        name_entry.insert(0, user_data[0])
        name_entry.pack(fill=tk.X, padx=20, pady=5)
        
        ttk.Label(dialog, text="Role:").pack(pady=(5, 0))
        role_entry = ttk.Entry(dialog)
        role_entry.insert(0, user_data[1] if user_data[1] else "")
        role_entry.pack(fill=tk.X, padx=20, pady=5)
        
        ttk.Label(dialog, text="Department:").pack(pady=(5, 0))
        dept_entry = ttk.Entry(dialog)
        dept_entry.insert(0, user_data[2] if user_data[2] else "")
        dept_entry.pack(fill=tk.X, padx=20, pady=5)
        
        ttk.Label(dialog, text="Notes:").pack(pady=(5, 0))
        notes_text = tk.Text(dialog, height=4)
        notes_text.insert("1.0", user_data[3] if user_data[3] else "")
        notes_text.pack(fill=tk.X, padx=20, pady=5)
        
        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(btn_frame, text="Cancel", command=dialog.destroy).pack(side=tk.RIGHT, padx=5)
        ttk.Button(btn_frame, text="Save", 
                  command=lambda: self.save_user_edit(
                      user_id,
                      name_entry.get(),
                      role_entry.get(),
                      dept_entry.get(),
                      notes_text.get("1.0", tk.END),
                      dialog
                  )).pack(side=tk.RIGHT, padx=5)

    def save_user_edit(self, user_id, name, role, department, notes, dialog):
        """Save edited user information"""
        if not name.strip():
            messagebox.showerror("Error", "Name cannot be empty")
            return
            
        try:
            cursor = self.db_conn.cursor()
            cursor.execute('''
                UPDATE users 
                SET name = ?, role = ?, department = ?, notes = ?
                WHERE id = ?
            ''', (name, role, department, notes, user_id))
            
            self.db_conn.commit()
            
            # Update in-memory data if needed
            if user_id in self.known_face_ids:
                index = self.known_face_ids.index(user_id)
                self.known_face_names[index] = name
                self.save_known_faces()
            
            self.load_users()
            self.status(f"User {name} updated")
            self.log(f"Updated user ID {user_id}")
            
            if self.config['enable_sound']:
                winsound.PlaySound("SystemExclamation", winsound.SND_ASYNC)
                
            dialog.destroy()
            
        except Exception as e:
            self.log_error(f"Failed to update user: {str(e)}")
            messagebox.showerror("Error", f"Failed to update user: {str(e)}")

    def delete_user(self):
        """Delete selected user with confirmation"""
        selected = self.user_tree.selection()
        if not selected:
            messagebox.showwarning("Warning", "Please select a user to delete")
            return
            
        user_id = self.user_tree.item(selected)['values'][0]
        user_name = self.user_tree.item(selected)['values'][1]
        
        if not messagebox.askyesno("Confirm Delete", f"Delete user {user_name} (ID: {user_id})?"):
            return
            
        try:
            cursor = self.db_conn.cursor()
            
            # Get image path before deleting
            cursor.execute('SELECT image_path FROM users WHERE id = ?', (user_id,))
            image_path = cursor.fetchone()[0]
            
            # Delete from database
            cursor.execute('DELETE FROM users WHERE id = ?', (user_id,))
            cursor.execute('DELETE FROM recognition_history WHERE user_id = ?', (user_id,))
            self.db_conn.commit()
            
            # Delete from known faces
            if user_id in self.known_face_ids:
                index = self.known_face_ids.index(user_id)
                del self.known_face_encodings[index]
                del self.known_face_ids[index]
                del self.known_face_names[index]
                self.save_known_faces()
            
            # Delete image file
            if image_path and os.path.exists(image_path):
                try:
                    os.remove(image_path)
                except Exception as e:
                    self.log_error(f"Failed to delete image: {str(e)}")
            
            self.load_users()
            self.load_recognition_history()
            self.status(f"User {user_name} deleted")
            self.log(f"Deleted user ID {user_id}")
            
            if self.config['enable_sound']:
                winsound.PlaySound("SystemHand", winsound.SND_ASYNC)
                
        except Exception as e:
            self.log_error(f"Failed to delete user: {str(e)}")
            messagebox.showerror("Error", f"Failed to delete user: {str(e)}")

    def export_users(self):
        """Export users to CSV file"""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
            title="Export Users To"
        )
        
        if not file_path:
            return
            
        try:
            cursor = self.db_conn.cursor()
            cursor.execute('''
                SELECT id, name, role, department, last_seen, first_registered, notes
                FROM users
                ORDER BY name
            ''')
            
            with open(file_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['ID', 'Name', 'Role', 'Department', 'Last Seen', 'First Registered', 'Notes'])
                writer.writerows(cursor.fetchall())
                
            self.status(f"Users exported to {file_path}")
            self.log(f"Exported users to {file_path}")
            
            if self.config['enable_sound']:
                winsound.PlaySound("SystemQuestion", winsound.SND_ASYNC)
                
        except Exception as e:
            self.log_error(f"Export failed: {str(e)}")
            messagebox.showerror("Export Error", f"Failed to export users: {str(e)}")

    def import_users(self):
        """Import users from CSV file"""
        file_path = filedialog.askopenfilename(
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
            title="Select Users CSV File"
        )
        
        if not file_path:
            return
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                
                cursor = self.db_conn.cursor()
                imported_count = 0
                
                for row in reader:
                    if len(row) < 7:  # Skip incomplete rows
                        continue
                        
                    try:
                        cursor.execute('''
                            INSERT OR REPLACE INTO users 
                            (id, name, role, department, last_seen, first_registered, notes)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        ''', row)
                        imported_count += 1
                    except sqlite3.Error:
                        continue
                        
                self.db_conn.commit()
                
            self.load_users()
            self.status(f"Imported {imported_count} users")
            self.log(f"Imported {imported_count} users from {file_path}")
            
            if self.config['enable_sound']:
                winsound.PlaySound("SystemQuestion", winsound.SND_ASYNC)
                
            messagebox.showinfo("Import Complete", f"Successfully imported {imported_count} users")
            
        except Exception as e:
            self.log_error(f"Import failed: {str(e)}")
            messagebox.showerror("Import Error", f"Failed to import users: {str(e)}")

    def show_docs(self):
        """Open documentation in browser"""
        webbrowser.open("https://github.com/yourusername/face-recognition-app/wiki")

    def show_about(self):
        """Show about dialog"""
        about_window = tk.Toplevel(self.root)
        about_window.title("About Professional Face Recognition")
        about_window.geometry("400x300")
        
        ttk.Label(about_window, text="Professional Face Recognition", font=('Arial', 14)).pack(pady=20)
        ttk.Label(about_window, text="Version 2.0").pack()
        ttk.Label(about_window, text="Developed by Your Name").pack(pady=10)
        ttk.Label(about_window, text="© 2023 All Rights Reserved").pack()
        
        ttk.Button(about_window, text="OK", command=about_window.destroy).pack(pady=20)

    def log(self, message):
        """Log message to console and UI"""
        if self.config['enable_logging']:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = f"[{timestamp}] {message}\n"
            
            self.log_text.config(state=tk.NORMAL)
            self.log_text.insert(tk.END, log_entry)
            self.log_text.config(state=tk.DISABLED)
            self.log_text.see(tk.END)
            
            print(log_entry, end='')

    def log_error(self, message):
        """Log error message with special formatting"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] ERROR: {message}\n"
        
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, log_entry, 'error')
        self.log_text.tag_config('error', foreground='red')
        self.log_text.config(state=tk.DISABLED)
        self.log_text.see(tk.END)
        
        print(log_entry, end='')

    def status(self, message):
        """Update status bar"""
        self.status_bar.config(text=message)
        self.log(f"Status: {message}")

    def start_background_tasks(self):
        """Start periodic background tasks"""
        self.update_frame()
        
        # Auto-save every 5 minutes
        self.root.after(self.config['auto_save_interval'] * 1000, self.auto_save)

    def auto_save(self):
        """Periodic auto-save of data"""
        try:
            self.save_known_faces()
            self.save_config()
            self.log("Auto-save completed")
        except Exception as e:
            self.log_error(f"Auto-save failed: {str(e)}")
            
        # Schedule next auto-save
        self.root.after(self.config['auto_save_interval'] * 1000, self.auto_save)

    def on_close(self):
        """Cleanup before closing"""
        self.recognizing = False
        self.is_capturing = False
        
        if hasattr(self, 'recognition_thread') and self.recognition_thread.is_alive():
            self.recognition_thread.join(timeout=1)
        
        if self.cap and self.cap.isOpened():
            self.cap.release()
            
        if hasattr(self, 'db_conn'):
            self.db_conn.close()
            
        self.save_config()
        self.save_known_faces()
        
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = ProfessionalFaceRecognitionApp(root)
    root.mainloop()
