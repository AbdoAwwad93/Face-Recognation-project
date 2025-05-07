import face_recognition
import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import threading
import time
import json
from datetime import datetime
import pickle
from PIL import Image, ImageTk
import io
import base64
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Set custom theme colors
PRIMARY_COLOR = "#4CAF50"  # Fresh Green
SECONDARY_COLOR = "#2196F3"  # Bright Blue
ACCENT_COLOR = "#FF9800"  # Warm Orange
BG_COLOR = "#FFFFFF"  # Pure White
TEXT_COLOR = "#333333"  # Dark Gray
SUCCESS_COLOR = "#4CAF50"  # Green
ERROR_COLOR = "#F44336"  # Red
WARNING_COLOR = "#FFC107"  # Amber
INFO_COLOR = "#00BCD4"  # Cyan

class ModernUI:
    """Class for modern UI elements and styles"""
    
    @staticmethod
    def create_rounded_button(parent, text, command, width=15, height=2, bg_color=PRIMARY_COLOR, 
                             hover_color=SECONDARY_COLOR, fg_color="white", corner_radius=10):
        """Create a modern rounded button with hover effect"""
        frame = tk.Frame(parent, bg=parent["bg"])
        
        def on_enter(e):
            button["background"] = hover_color
            
        def on_leave(e):
            button["background"] = bg_color
        
        button = tk.Button(
            frame, 
            text=text, 
            command=command,
            width=width,
            height=height,
            bg=bg_color,
            fg=fg_color,
            relief=tk.FLAT,
            font=("Helvetica", 10, "bold"),
            cursor="hand2"
        )
        
        button.pack(padx=5, pady=5)
        button.bind("<Enter>", on_enter)
        button.bind("<Leave>", on_leave)
        
        return frame
    
    @staticmethod
    def create_title(parent, text, font_size=16):
        """Create a modern title with underline effect"""
        frame = tk.Frame(parent, bg=parent["bg"])
        
        title = tk.Label(
            frame,
            text=text,
            font=("Helvetica", font_size, "bold"),
            fg=PRIMARY_COLOR,
            bg=parent["bg"]
        )
        title.pack(pady=(10, 5))
        
        # Add underline
        separator = ttk.Separator(frame, orient="horizontal")
        separator.pack(fill="x", padx=20)
        
        return frame
    
    @staticmethod
    def create_card(parent, padding=15, bg_color="white"):
        """Create a card-like frame with shadow effect"""
        # Main frame
        outer_frame = tk.Frame(parent, bg=parent["bg"], padx=2, pady=2)
        
        # Inner frame (card)
        card = tk.Frame(
            outer_frame,
            bg=bg_color,
            padx=padding,
            pady=padding,
            relief=tk.RAISED,
            bd=1
        )
        card.pack(fill="both", expand=True)
        
        return outer_frame, card
    
    @staticmethod
    def apply_theme(root):
        """Apply custom theme to ttk widgets"""
        style = ttk.Style(root)
        
        # Configure ttk theme
        style.configure("TFrame", background=BG_COLOR)
        style.configure("TLabel", background=BG_COLOR, foreground=TEXT_COLOR)
        style.configure("TButton", background=PRIMARY_COLOR, foreground="white", font=("Helvetica", 10))
        style.map("TButton", 
                 background=[("active", SECONDARY_COLOR), ("disabled", "#cccccc")],
                 foreground=[("disabled", "#999999")])
        
        # Configure custom styles
        style.configure("Success.TLabel", foreground=SUCCESS_COLOR)
        style.configure("Error.TLabel", foreground=ERROR_COLOR)
        style.configure("Title.TLabel", font=("Helvetica", 16, "bold"), foreground=PRIMARY_COLOR)
        style.configure("Subtitle.TLabel", font=("Helvetica", 12, "bold"), foreground=PRIMARY_COLOR)
        
        return style


class FaceLoginSystem:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.team_face_encodings = []  # Separate encodings for team members
        self.team_face_names = []      # Separate names for team members
        self.user_data = {}
        self.current_user = None
        self.is_running = False
        self.cap = None
        self.data_dir = "face_login_data"
        self.encodings_file = os.path.join(self.data_dir, "encodings.pkl")
        self.team_encodings_file = os.path.join(self.data_dir, "team_encodings.pkl")
        self.users_file = os.path.join(self.data_dir, "users.json")
        self.faces_dir = os.path.join(self.data_dir, "faces")
        
        # New standalone training directory
        self.training_dir = "training_images"  # Changed to standalone directory
        self.team_members_dir = os.path.join(self.training_dir, "team_members")
        
        # Create data directory if it doesn't exist
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        if not os.path.exists(self.faces_dir):
            os.makedirs(self.faces_dir)
        if not os.path.exists(self.training_dir):
            os.makedirs(self.training_dir)
        if not os.path.exists(self.team_members_dir):
            os.makedirs(self.team_members_dir)
        
        # Load existing data
        self.load_data()
        self.load_team_data()

    def load_data(self):
        """Load user data and face encodings"""
        # Load user data
        if os.path.exists(self.users_file):
            try:
                with open(self.users_file, 'r') as f:
                    self.user_data = json.load(f)
                print(f"[INFO] Loaded {len(self.user_data)} user profiles")
            except Exception as e:
                print(f"[ERROR] Failed to load user data: {e}")
                self.user_data = {}
        
        # Load face encodings
        if os.path.exists(self.encodings_file):
            try:
                with open(self.encodings_file, 'rb') as f:
                    data = pickle.load(f)
                    self.known_face_encodings = data['encodings']
                    self.known_face_names = data['names']
                print(f"[INFO] Loaded {len(self.known_face_encodings)} face encodings")
            except Exception as e:
                print(f"[ERROR] Failed to load face encodings: {e}")
                self.known_face_encodings = []
                self.known_face_names = []

    def save_data(self):
        """Save user data and face encodings"""
        # Save user data
        try:
            with open(self.users_file, 'w') as f:
                json.dump(self.user_data, f, indent=4)
        except Exception as e:
            print(f"[ERROR] Failed to save user data: {e}")
        
        # Save face encodings
        try:
            with open(self.encodings_file, 'wb') as f:
                data = {
                    'encodings': self.known_face_encodings,
                    'names': self.known_face_names
                }
                pickle.dump(data, f)
        except Exception as e:
            print(f"[ERROR] Failed to save face encodings: {e}")

    def register_new_user(self, username, password, additional_info=None):
        """Register a new user with password and optional face capture"""
        if username in self.user_data:
            return False, "Username already exists"
        
        # Create user profile with password
        user_profile = {
            "password": password,  # In a real app, this should be hashed
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "login_history": [],
            "info": additional_info or {}
        }
        self.user_data[username] = user_profile
        
        # Save data
        self.save_data()
        
        return True, "User registered successfully"

    def verify_password(self, username, password):
        """Verify user password"""
        if username not in self.user_data:
            return False, "Username not found"
        
        if self.user_data[username]["password"] == password:  # In a real app, compare hashed passwords
            return True, "Password verified"
        return False, "Invalid password"

    def register_face(self, username, face_encoding, face_image):
        """Register face for an existing user"""
        if username not in self.user_data:
            return False, "Username not found"
        
        try:
            # Save face image
            face_path = os.path.join(self.faces_dir, f"{username}.jpg")
            cv2.imwrite(face_path, face_image)
            
            # Add face encoding and username
            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(username)
            
            # Update user profile
            self.user_data[username]["has_face"] = True
            
            # Save data
            self.save_data()
            
            return True, "Face registered successfully"
        except Exception as e:
            return False, f"Failed to register face: {str(e)}"

    def start_login(self):
        """Start the face recognition login process"""
        if not self.known_face_encodings:
            return False, "No registered users found. Please register first."
        
        self.is_running = True
        self.current_user = None
        
        # Start recognition in a separate thread
        login_thread = threading.Thread(target=self.login_recognition_loop)
        login_thread.daemon = True
        login_thread.start()
        
        return True, "Login process started"

    def login_recognition_loop(self):
        """Face recognition loop for login"""
        try:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            print("[INFO] Starting face recognition login. Press 'q' to quit.")
            
            while self.is_running:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Create a copy for display
                display_frame = frame.copy()
                
                # Resize frame for faster processing
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                
                # Find faces
                face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
                
                if face_locations:
                    # Get face encodings
                    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                    
                    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                        # Scale back to original size
                        top *= 4
                        right *= 4
                        bottom *= 4
                        left *= 4
                        
                        # Match against known faces
                        matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.5)
                        name = "Unknown"
                        
                        if True in matches:
                            # Find the best match
                            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                            best_match_index = np.argmin(face_distances)
                            if matches[best_match_index]:
                                name = self.known_face_names[best_match_index]
                                
                                # Login successful
                                print(f"[INFO] Login successful: {name}")
                                self.current_user = name
                                
                                # Record login
                                if name in self.user_data:
                                    self.user_data[name]["login_history"].append(
                                        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                    )
                                    self.save_data()
                                
                                self.is_running = False
                                break
                        
                        # Draw rectangle with name
                        cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                        cv2.rectangle(display_frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                        cv2.putText(display_frame, name, (left + 6, bottom - 6), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
                        
                        # If face is unknown, close the window
                        if name == "Unknown":
                            cv2.destroyAllWindows()
                            self.is_running = False
                            break
                
                # Display the frame
                cv2.imshow("Face Login", display_frame)
                
                # Check for key press
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.is_running = False
                    break
                
                # Exit if login successful
                if self.current_user:
                    break
                    
        except Exception as e:
            print(f"[ERROR] Exception in login recognition loop: {e}")
        finally:
            # Clean up resources
            if self.cap is not None:
                self.cap.release()
            cv2.destroyAllWindows()

    def stop_recognition(self):
        """Stop the recognition process"""
        self.is_running = False
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()

    def get_user_face_image(self, username):
        """Get the user's face image as a PhotoImage"""
        face_path = os.path.join(self.faces_dir, f"{username}.jpg")
        if os.path.exists(face_path):
            try:
                img = Image.open(face_path)
                img = img.resize((150, 150), Image.LANCZOS)
                return ImageTk.PhotoImage(img)
            except Exception as e:
                print(f"[ERROR] Failed to load user image: {e}")
        return None

    @staticmethod
    def center_window(window, width, height):
        """Center a window on the screen"""
        screen_width = window.winfo_screenwidth()
        screen_height = window.winfo_screenheight()
        x = (screen_width // 2) - (width // 2)
        y = (screen_height // 2) - (height // 2)
        window.geometry(f"{width}x{height}+{x}+{y}")

    def add_training_image(self, username, face_image, image_name=None):
        """Add a training image for a user to improve recognition accuracy"""
        if username not in self.user_data:
            return False, "Username not found"
        
        try:
            # Create user training directory if it doesn't exist
            user_training_dir = os.path.join(self.training_dir, username)
            if not os.path.exists(user_training_dir):
                os.makedirs(user_training_dir)
            
            # Generate image name if not provided
            if image_name is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                image_name = f"training_{timestamp}.jpg"
            
            # Save the image
            image_path = os.path.join(user_training_dir, image_name)
            cv2.imwrite(image_path, face_image)
            
            # Get face encoding
            rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_image, model="hog")
            
            if not face_locations:
                return False, "No face detected in the image"
            
            face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
            if not face_encodings:
                return False, "Could not encode face"
            
            # Add to known encodings
            self.known_face_encodings.append(face_encodings[0])
            self.known_face_names.append(username)
            
            # Save updated encodings
            self.save_data()
            
            return True, f"Training image added successfully: {image_name}"
            
        except Exception as e:
            return False, f"Failed to add training image: {str(e)}"
    
    def get_training_images(self, username):
        """Get list of training images for a user"""
        if username not in self.user_data:
            return []
        
        user_training_dir = os.path.join(self.training_dir, username)
        if not os.path.exists(user_training_dir):
            return []
        
        return [f for f in os.listdir(user_training_dir) if f.endswith(('.jpg', '.jpeg', '.png','.JPG'))]
    
    def delete_training_image(self, username, image_name):
        """Delete a training image for a user"""
        if username not in self.user_data:
            return False, "Username not found"
        
        try:
            image_path = os.path.join(self.training_dir, username, image_name)
            if not os.path.exists(image_path):
                return False, "Image not found"
            
            os.remove(image_path)
            return True, f"Training image deleted: {image_name}"
            
        except Exception as e:
            return False, f"Failed to delete training image: {str(e)}"
    
    def reload_encodings_from_training(self):
        """Reload all face encodings from training images"""
        self.known_face_encodings = []
        self.known_face_names = []
        
        # Process each user's training directory
        for username in os.listdir(self.training_dir):
            user_dir = os.path.join(self.training_dir, username)
            if not os.path.isdir(user_dir) or username == "team_members":  # Skip team_members directory
                continue
                
            # Process each image in the user's directory
            for image_name in os.listdir(user_dir):
                if not image_name.endswith(('.jpg', '.jpeg', '.png','.JPG')):
                    continue
                    
                image_path = os.path.join(user_dir, image_name)
                try:
                    # Load and process the image
                    image = face_recognition.load_image_file(image_path)
                    face_locations = face_recognition.face_locations(image, model="hog")
                    
                    if face_locations:
                        face_encodings = face_recognition.face_encodings(image, face_locations)
                        if face_encodings:
                            # Add each encoding
                            for encoding in face_encodings:
                                self.known_face_encodings.append(encoding)
                                self.known_face_names.append(username)
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
        
        # Save the updated encodings
        self.save_data()
        return len(self.known_face_encodings)

    def load_team_data(self):
        """Load team member face encodings"""
        # Load team encodings
        if os.path.exists(self.team_encodings_file):
            try:
                with open(self.team_encodings_file, 'rb') as f:
                    data = pickle.load(f)
                    self.team_face_encodings = data['encodings']
                    self.team_face_names = data['names']
                print(f"[INFO] Loaded {len(self.team_face_encodings)} team member face encodings")
            except Exception as e:
                print(f"[ERROR] Failed to load team encodings: {e}")
                self.team_face_encodings = []
                self.team_face_names = []

    def save_team_data(self):
        """Save team member face encodings"""
        try:
            with open(self.team_encodings_file, 'wb') as f:
                data = {
                    'encodings': self.team_face_encodings,
                    'names': self.team_face_names
                }
                pickle.dump(data, f)
        except Exception as e:
            print(f"[ERROR] Failed to save team encodings: {e}")

    def reload_team_encodings(self):
        """Reload all face encodings from team member training images"""
        self.team_face_encodings = []
        self.team_face_names = []
        
        # Process each team member's directory
        for member_name in os.listdir(self.team_members_dir):
            member_dir = os.path.join(self.team_members_dir, member_name)
            if not os.path.isdir(member_dir):
                continue
                
            # Process each image in the member's directory
            for image_name in os.listdir(member_dir):
                if not image_name.endswith(('.jpg', '.jpeg', '.png','')):
                    continue
                    
                image_path = os.path.join(member_dir, image_name)
                try:
                    # Load and process the image
                    image = face_recognition.load_image_file(image_path)
                    face_locations = face_recognition.face_locations(image, model="hog")
                    
                    if face_locations:
                        face_encodings = face_recognition.face_encodings(image, face_locations)
                        if face_encodings:
                            # Add each encoding
                            for encoding in face_encodings:
                                self.team_face_encodings.append(encoding)
                                self.team_face_names.append(member_name)
                except Exception as e:
                    print(f"Error processing team member image {image_path}: {e}")
        
        # Save the updated team encodings
        self.save_team_data()
        return len(self.team_face_encodings)


class EnhancedLoginApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Login System")
        self.face_system = FaceLoginSystem()
        
        # Set window size and center it
        self.face_system.center_window(root, 400, 500)
        
        # Set background color
        self.root.configure(bg=BG_COLOR)
        
        # Apply modern theme
        self.style = ModernUI.apply_theme(root)
        
        # Create main container
        self.main_container = tk.Frame(root, bg=BG_COLOR, padx=20, pady=20)
        self.main_container.pack(fill=tk.BOTH, expand=True)
        
        # Create header
        self.create_header()
        
        # Create login form
        self.create_login_form()
        
        # Create footer
        self.create_footer()

    def create_header(self):
        """Create header with title"""
        header_frame = tk.Frame(self.main_container, bg=BG_COLOR)
        header_frame.pack(fill="x", pady=(0, 20))
        
        title = tk.Label(
            header_frame,
            text="Login System",
            font=("Helvetica", 24, "bold"),
            fg=PRIMARY_COLOR,
            bg=BG_COLOR
        )
        title.pack()

    def create_login_form(self):
        """Create the login form"""
        # Create login card
        login_outer, login_card = ModernUI.create_card(self.main_container)
        login_outer.pack(fill="both", expand=True)
        
        # Username field
        username_frame = tk.Frame(login_card, bg="white")
        username_frame.pack(fill="x", pady=10)
        
        username_label = tk.Label(
            username_frame,
            text="Username:",
            font=("Helvetica", 10, "bold"),
            fg=TEXT_COLOR,
            bg="white"
        )
        username_label.pack(anchor="w")
        
        self.username_entry = tk.Entry(
            username_frame,
            font=("Helvetica", 12),
            width=30
        )
        self.username_entry.pack(fill="x", pady=(5, 0))
        
        # Password field
        password_frame = tk.Frame(login_card, bg="white")
        password_frame.pack(fill="x", pady=10)
        
        password_label = tk.Label(
            password_frame,
            text="Password:",
            font=("Helvetica", 10, "bold"),
            fg=TEXT_COLOR,
            bg="white"
        )
        password_label.pack(anchor="w")
        
        self.password_entry = tk.Entry(
            password_frame,
            font=("Helvetica", 12),
            width=30,
            show="*"
        )
        self.password_entry.pack(fill="x", pady=(5, 0))
        
        # Login buttons
        buttons_frame = tk.Frame(login_card, bg="white")
        buttons_frame.pack(fill="x", pady=20)
        
        # Password login button
        password_login_btn = ModernUI.create_rounded_button(
            buttons_frame,
            "Login with Password",
            self.login_with_password,
            width=20,
            height=1
        )
        password_login_btn.pack(pady=5)
        
        # Face login button
        face_login_btn = ModernUI.create_rounded_button(
            buttons_frame,
            "Login with Face",
            self.login_with_face,
            width=20,
            height=1,
            bg_color=SECONDARY_COLOR,
            hover_color=PRIMARY_COLOR
        )
        face_login_btn.pack(pady=5)
        
        # Register button
        register_btn = ModernUI.create_rounded_button(
            buttons_frame,
            "Register New User",
            self.register,
            width=20,
            height=1,
            bg_color=ACCENT_COLOR,
            hover_color=self._darken_color(ACCENT_COLOR)
        )
        register_btn.pack(pady=5)
        
        # Status message
        self.status_label = tk.Label(
            login_card,
            text="Welcome! Please login or register.",
            font=("Helvetica", 10),
            fg=TEXT_COLOR,
            bg="white",
            wraplength=350
        )
        self.status_label.pack(pady=10)

    def create_footer(self):
        """Create footer with exit button"""
        footer_frame = tk.Frame(self.main_container, bg=BG_COLOR)
        footer_frame.pack(fill="x", pady=(20, 0))
        
        exit_btn = ModernUI.create_rounded_button(
            footer_frame,
            "Exit",
            self.exit_app,
            width=10,
            height=1,
            bg_color=ERROR_COLOR,
            hover_color="#d32f2f"
        )
        exit_btn.pack(side="right")

    def login_with_password(self):
        """Handle login with password"""
        username = self.username_entry.get().strip()
        password = self.password_entry.get().strip()
        
        if not username or not password:
            self.status_label.config(text="Please enter both username and password", fg=ERROR_COLOR)
            return
        
        success, message = self.face_system.verify_password(username, password)
        if success:
            self.face_system.current_user = username
            self.status_label.config(text=f"Login successful: {username}", fg=SUCCESS_COLOR)
            self.root.after(1000, self.show_dashboard)
        else:
            self.status_label.config(text=message, fg=ERROR_COLOR)

    def login_with_face(self):
        """Handle login with face recognition"""
        if not self.face_system.known_face_encodings:
            self.status_label.config(text="No registered users found. Please register first.", fg=ERROR_COLOR)
            return
        
        # Get username from the existing textbox
        username = self.username_entry.get().strip()
        if not username:
            self.status_label.config(text="Please enter your username first", fg=ERROR_COLOR)
            return
        
        # Check if username exists
        if username not in self.face_system.user_data:
            self.status_label.config(text=f"Username '{username}' not found", fg=ERROR_COLOR)
            return
        
        self.status_label.config(text="Starting face recognition...", fg=PRIMARY_COLOR)
        
        # Create authentication dialog
        auth_dialog = tk.Toplevel(self.root)
        auth_dialog.title("Face Authentication")
        auth_dialog.configure(bg=BG_COLOR)
        self.face_system.center_window(auth_dialog, 400, 300)  # Made smaller since we won't show auth fields
        auth_dialog.grab_set()
        
        # Create content
        content_frame = tk.Frame(auth_dialog, bg=BG_COLOR, padx=20, pady=20)
        content_frame.pack(fill="both", expand=True)
        
        # Title
        title_label = tk.Label(
            content_frame,
            text="Face Authentication",
            font=("Helvetica", 18, "bold"),
            fg=PRIMARY_COLOR,
            bg=BG_COLOR
        )
        title_label.pack(pady=(0, 20))
        
        # Username display
        username_label = tk.Label(
            content_frame,
            text=f"Verifying face for: {username}",
            font=("Helvetica", 12, "bold"),
            fg=TEXT_COLOR,
            bg=BG_COLOR
        )
        username_label.pack(pady=(0, 10))
        
        # Status message
        status_label = tk.Label(
            content_frame,
            text="Looking for your face...",
            font=("Helvetica", 10),
            fg=TEXT_COLOR,
            bg=BG_COLOR,
            wraplength=350
        )
        status_label.pack(pady=10)
        
        # Buttons frame
        buttons_frame = tk.Frame(content_frame, bg=BG_COLOR)
        buttons_frame.pack(fill="x", pady=10)
        
        def complete_login(dialog):
            """Complete the login process"""
            dialog.destroy()
            self.root.after(1000, self.show_dashboard)
        
        def show_error_and_close(message):
            """Show error message and close dialog after delay"""
            status_label.config(text=message, fg=ERROR_COLOR)
            auth_dialog.after(2000, auth_dialog.destroy)
        
        # Start face verification
        success, message = self.face_system.start_login()
        if not success:
            status_label.config(text=message, fg=ERROR_COLOR)
            return
        
        # Check login status periodically
        def check_login_status():
            if not self.face_system.is_running:
                if self.face_system.current_user:
                    # Verify that the recognized face matches the provided username
                    if self.face_system.current_user == username:
                        status_label.config(text=f"Login successful: {username}", fg=SUCCESS_COLOR)
                        auth_dialog.after(1000, lambda: complete_login(auth_dialog))
                    else:
                        # Face recognized but doesn't match the provided username
                        show_error_and_close(f"Wrong username or face missmatch.")
                else:
                    # Show error message and close dialog
                    show_error_and_close("Authentication failed. Face not recognized.")
            else:
                # Login process still running, check again after 500ms
                auth_dialog.after(500, check_login_status)
        
        check_login_status()
        
        # Handle window close
        auth_dialog.protocol("WM_DELETE_WINDOW", lambda: self.face_system.stop_recognition())

    def register(self):
        """Handle register button click"""
        # Create registration dialog
        register_dialog = tk.Toplevel(self.root)
        register_dialog.title("Register New User")
        register_dialog.configure(bg=BG_COLOR)
        self.face_system.center_window(register_dialog, 400, 600)  # Made taller for face registration
        register_dialog.grab_set()
        
        # Create content
        content_frame = tk.Frame(register_dialog, bg=BG_COLOR, padx=20, pady=20)
        content_frame.pack(fill="both", expand=True)
        
        # Title
        title_label = tk.Label(
            content_frame,
            text="Register New User",
            font=("Helvetica", 18, "bold"),
            fg=PRIMARY_COLOR,
            bg=BG_COLOR
        )
        title_label.pack(pady=(0, 20))
        
        # Username field
        username_frame = tk.Frame(content_frame, bg=BG_COLOR)
        username_frame.pack(fill="x", pady=10)
        
        username_label = tk.Label(
            username_frame,
            text="Username:",
            font=("Helvetica", 10, "bold"),
            fg=TEXT_COLOR,
            bg=BG_COLOR
        )
        username_label.pack(anchor="w")
        
        username_entry = tk.Entry(
            username_frame,
            font=("Helvetica", 12),
            width=30
        )
        username_entry.pack(fill="x", pady=(5, 0))
        
        # Password field
        password_frame = tk.Frame(content_frame, bg=BG_COLOR)
        password_frame.pack(fill="x", pady=10)
        
        password_label = tk.Label(
            password_frame,
            text="Password:",
            font=("Helvetica", 10, "bold"),
            fg=TEXT_COLOR,
            bg=BG_COLOR
        )
        password_label.pack(anchor="w")
        
        password_entry = tk.Entry(
            password_frame,
            font=("Helvetica", 12),
            width=30,
            show="*"
        )
        password_entry.pack(fill="x", pady=(5, 0))
        
        # Confirm Password field
        confirm_frame = tk.Frame(content_frame, bg=BG_COLOR)
        confirm_frame.pack(fill="x", pady=10)
        
        confirm_label = tk.Label(
            confirm_frame,
            text="Confirm Password:",
            font=("Helvetica", 10, "bold"),
            fg=TEXT_COLOR,
            bg=BG_COLOR
        )
        confirm_label.pack(anchor="w")
        
        confirm_entry = tk.Entry(
            confirm_frame,
            font=("Helvetica", 12),
            width=30,
            show="*"
        )
        confirm_entry.pack(fill="x", pady=(5, 0))
        
        # Face registration status
        face_status_frame = tk.Frame(content_frame, bg=BG_COLOR)
        face_status_frame.pack(fill="x", pady=10)
        
        face_status_label = tk.Label(
            face_status_frame,
            text="Face not registered yet",
            font=("Helvetica", 10),
            fg=ERROR_COLOR,
            bg=BG_COLOR
        )
        face_status_label.pack(side="left")
        
        # Store face data
        face_data = {"registered": False}
        
        # Status message
        status_label = tk.Label(
            content_frame,
            text="",
            font=("Helvetica", 10),
            fg=TEXT_COLOR,
            bg=BG_COLOR,
            wraplength=350
        )
        status_label.pack(pady=10)
        
        # Buttons frame
        buttons_frame = tk.Frame(content_frame, bg=BG_COLOR)
        buttons_frame.pack(fill="x", pady=10)
        
        def on_register_face():
            username = username_entry.get().strip()
            
            # Basic validation
            if not username:
                status_label.config(text="Please enter a username first", fg=ERROR_COLOR)
                return
            
            if not username.isalnum():
                status_label.config(text="Username must contain only letters and numbers", fg=ERROR_COLOR)
                return
            
            if username in self.face_system.user_data:
                status_label.config(text="Username already exists", fg=ERROR_COLOR)
                return
            
            # Initialize webcam and capture face
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                status_label.config(text="Failed to initialize webcam", fg=ERROR_COLOR)
                return
            
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            face_detected = False
            face_encoding = None
            face_image = None
            
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                        
                    display_frame = frame.copy()
                    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                    
                    face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
                    
                    for (top, right, bottom, left) in face_locations:
                        top *= 4
                        right *= 4
                        bottom *= 4
                        left *= 4
                        cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    
                    cv2.putText(display_frame, "Press 'c' to capture face", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(display_frame, "Press 'q' to cancel", (10, 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    cv2.imshow("Register Face", display_frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    
                    if key == ord('c'):
                        if face_locations:
                            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                            if face_encodings:
                                face_encoding = face_encodings[0]
                                face_image = frame.copy()
                                face_detected = True
                                break
                        else:
                            status_label.config(text="No face detected. Please position your face in the frame.", fg=ERROR_COLOR)
                    
                    elif key == ord('q'):
                        break
            
            finally:
                cap.release()
                cv2.destroyAllWindows()
            
            if face_detected and face_encoding is not None:
                face_data["encoding"] = face_encoding
                face_data["image"] = face_image
                face_data["registered"] = True
                face_status_label.config(text="Face registered successfully!", fg=SUCCESS_COLOR)
                status_label.config(text="Face captured successfully! You can now submit the registration.", fg=SUCCESS_COLOR)
                submit_btn.config(state=tk.NORMAL)  # Enable submit button
            else:
                status_label.config(text="Face registration failed. Please try again.", fg=ERROR_COLOR)
        
        def on_submit():
            username = username_entry.get().strip()
            password = password_entry.get().strip()
            confirm = confirm_entry.get().strip()
            
            if not username or not password or not confirm:
                status_label.config(text="Please fill in all fields", fg=ERROR_COLOR)
                return
            
            if password != confirm:
                status_label.config(text="Passwords do not match", fg=ERROR_COLOR)
                return
            
            if not face_data["registered"]:
                status_label.config(text="Please register your face first", fg=ERROR_COLOR)
                return
            
            try:
                # Register user with password
                success, message = self.face_system.register_new_user(username, password)
                if not success:
                    status_label.config(text=message, fg=ERROR_COLOR)
                    return
                
                # Register face
                success, message = self.face_system.register_face(
                    username, 
                    face_data["encoding"], 
                    face_data["image"]
                )
                
                if success:
                    status_label.config(text="Registration successful! You can now login with password or face.", fg=SUCCESS_COLOR)
                    self.root.after(2000, register_dialog.destroy)
                else:
                    status_label.config(text=message, fg=ERROR_COLOR)
                
            except Exception as e:
                status_label.config(text=f"Failed to save user data: {str(e)}", fg=ERROR_COLOR)
        
        # Register Face button
        register_face_btn = ModernUI.create_rounded_button(
            buttons_frame,
            "Register Face",
            on_register_face,
            width=15,
            height=1,
            bg_color=SECONDARY_COLOR,
            hover_color=PRIMARY_COLOR
        )
        register_face_btn.pack(side="left", padx=(0, 5))
        
        # Submit button (initially disabled)
        submit_btn = ModernUI.create_rounded_button(
            buttons_frame,
            "Register",
            on_submit,
            width=15,
            height=1,
            bg_color=PRIMARY_COLOR,
            hover_color=self._darken_color(PRIMARY_COLOR)
        )
        submit_btn.pack(side="left", padx=5)
        
        # Cancel button
        cancel_btn = ModernUI.create_rounded_button(
            buttons_frame,
            "Cancel",
            register_dialog.destroy,
            width=15,
            height=1,
            bg_color=ERROR_COLOR,
            hover_color=self._darken_color(ERROR_COLOR)
        )
        cancel_btn.pack(side="right")

    def show_dashboard(self):
        """Show dashboard after successful login"""
        # Hide main window
        self.root.withdraw()
        
        # Create dashboard window
        dashboard = tk.Toplevel(self.root)
        dashboard.title(f"Welcome {self.face_system.current_user}")
        dashboard.configure(bg=BG_COLOR)
        self.face_system.center_window(dashboard, 500, 400)  # Made larger to accommodate more buttons
        
        # Create main container
        main_frame = tk.Frame(dashboard, bg=BG_COLOR, padx=20, pady=20)
        main_frame.pack(fill="both", expand=True)
        
        # Welcome message
        welcome_label = tk.Label(
            main_frame,
            text=f"Welcome, {self.face_system.current_user}!",
            font=("Helvetica", 20, "bold"),
            fg=PRIMARY_COLOR,
            bg=BG_COLOR
        )
        welcome_label.pack(pady=20)
        
        # Buttons frame
        buttons_frame = tk.Frame(main_frame, bg=BG_COLOR)
        buttons_frame.pack(fill="x", pady=20)
        
        # Recognize Team Members button
        recognize_team_btn = ModernUI.create_rounded_button(
            buttons_frame,
            "Recognize Team Members",
            lambda: self.recognize_team_members(dashboard),
            width=20,
            height=1,
            bg_color=SECONDARY_COLOR,
            hover_color=PRIMARY_COLOR
        )
        recognize_team_btn.pack(pady=10)
        
        # Logout button
        logout_btn = ModernUI.create_rounded_button(
            buttons_frame,
            "Logout",
            lambda: self.logout(dashboard),
            width=15,
            height=1,
            bg_color=ERROR_COLOR,
            hover_color=self._darken_color(ERROR_COLOR)
        )
        logout_btn.pack(pady=10)
        
        # Handle window close
        dashboard.protocol("WM_DELETE_WINDOW", lambda: self.logout(dashboard))

    def recognize_team_members(self, parent_window):
        """Recognize team members using face recognition"""
        # First reload team encodings to ensure we have the latest data
        self.face_system.reload_team_encodings()
        
        if not self.face_system.team_face_encodings:
            messagebox.showwarning("No Team Members", "No team member images found. Please add team member images first.")
            return

        # Create recognition dialog
        recognition_dialog = tk.Toplevel(parent_window)
        recognition_dialog.title("Team Member Recognition")
        recognition_dialog.configure(bg=BG_COLOR)
        self.face_system.center_window(recognition_dialog, 600, 500)
        recognition_dialog.grab_set()
        
        # Create content
        content_frame = tk.Frame(recognition_dialog, bg=BG_COLOR, padx=20, pady=20)
        content_frame.pack(fill="both", expand=True)
        
        # Title
        title_label = tk.Label(
            content_frame,
            text="Team Member Recognition",
            font=("Helvetica", 18, "bold"),
            fg=PRIMARY_COLOR,
            bg=BG_COLOR
        )
        title_label.pack(pady=(0, 20))
        
        # Status message
        status_label = tk.Label(
            content_frame,
            text="Looking for team members...",
            font=("Helvetica", 10),
            fg=TEXT_COLOR,
            bg=BG_COLOR,
            wraplength=550
        )
        status_label.pack(pady=10)
        
        # Results frame
        results_frame = tk.Frame(content_frame, bg=BG_COLOR)
        results_frame.pack(fill="both", expand=True, pady=10)
        
        # Create a text widget to display results
        results_text = tk.Text(
            results_frame,
            wrap=tk.WORD,
            font=("Helvetica", 10),
            bg="white",
            fg=TEXT_COLOR,
            height=10,
            width=50
        )
        results_text.pack(fill="both", expand=True, pady=10)
        results_text.insert(tk.END, "Recognized team members will appear here...\n")
        
        # Buttons frame
        buttons_frame = tk.Frame(content_frame, bg=BG_COLOR)
        buttons_frame.pack(fill="x", pady=10)
        
        # Start recognition button
        start_btn = ModernUI.create_rounded_button(
            buttons_frame,
            "Start Recognition",
            lambda: start_recognition(),
            width=15,
            height=1,
            bg_color=PRIMARY_COLOR,
            hover_color=self._darken_color(PRIMARY_COLOR)
        )
        start_btn.pack(side="left", padx=5)
        
        # Close button
        close_btn = ModernUI.create_rounded_button(
            buttons_frame,
            "Close",
            recognition_dialog.destroy,
            width=15,
            height=1,
            bg_color=ERROR_COLOR,
            hover_color=self._darken_color(ERROR_COLOR)
        )
        close_btn.pack(side="right", padx=5)
        
        # Flag to control recognition
        is_running = [False]
        cap = [None]
        
        def start_recognition():
            """Start the team member recognition process"""
            if is_running[0]:
                return
                
            is_running[0] = True
            status_label.config(text="Recognition in progress...", fg=PRIMARY_COLOR)
            results_text.delete(1.0, tk.END)
            results_text.insert(tk.END, "Starting recognition...\n")
            
            # Start recognition in a separate thread
            recognition_thread = threading.Thread(target=recognition_loop)
            recognition_thread.daemon = True
            recognition_thread.start()
            
        def recognition_loop():
            """Face recognition loop for team members"""
            try:
                cap[0] = cv2.VideoCapture(0)
                cap[0].set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap[0].set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                
                print("[INFO] Starting team member recognition. Press 'q' to quit.")
                
                # Track recognized faces to avoid duplicates
                recognized_faces = set()
                
                while is_running[0]:
                    ret, frame = cap[0].read()
                    if not ret:
                        break
                    
                    # Create a copy for display
                    display_frame = frame.copy()
                    
                    # Resize frame for faster processing
                    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                    
                    # Find faces
                    face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
                    
                    if face_locations:
                        # Get face encodings
                        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                        
                        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                            # Scale back to original size
                            top *= 4
                            right *= 4
                            bottom *= 4
                            left *= 4
                            
                            # Match against known team faces only
                            matches = face_recognition.compare_faces(self.face_system.team_face_encodings, face_encoding, tolerance=0.5)
                            name = "Unknown"
                            
                            if True in matches:
                                # Find the best match
                                face_distances = face_recognition.face_distance(self.face_system.team_face_encodings, face_encoding)
                                best_match_index = np.argmin(face_distances)
                                if matches[best_match_index]:
                                    name = self.face_system.team_face_names[best_match_index]
                                    
                                    # Add to recognized faces if not already there
                                    if name not in recognized_faces:
                                        recognized_faces.add(name)
                                        # Update results in the main thread
                                        recognition_dialog.after(0, lambda n=name: update_results(n))
                            
                            # Draw rectangle with name
                            cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                            cv2.rectangle(display_frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                            cv2.putText(display_frame, name, (left + 6, bottom - 6), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
                    
                    # Display the frame
                    cv2.imshow("Team Member Recognition", display_frame)
                    
                    # Check for key press
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        is_running[0] = False
                        break
                        
            except Exception as e:
                print(f"[ERROR] Exception in team recognition loop: {e}")
                recognition_dialog.after(0, lambda: status_label.config(
                    text=f"Error: {str(e)}", fg=ERROR_COLOR))
            finally:
                # Clean up resources
                if cap[0] is not None:
                    cap[0].release()
                cv2.destroyAllWindows()
                is_running[0] = False
                recognition_dialog.after(0, lambda: status_label.config(
                    text="Recognition stopped", fg=TEXT_COLOR))
        
        def update_results(name):
            """Update the results text widget with a recognized team member"""
            current_time = datetime.now().strftime("%H:%M:%S")
            results_text.insert(tk.END, f"[{current_time}] Recognized: {name}\n")
            results_text.see(tk.END)  # Scroll to the end
        
        # Handle window close
        recognition_dialog.protocol("WM_DELETE_WINDOW", lambda: stop_recognition())
        
        def stop_recognition():
            """Stop the recognition process"""
            is_running[0] = False
            if cap[0] is not None:
                cap[0].release()
            cv2.destroyAllWindows()
            recognition_dialog.destroy()

    def logout(self, dashboard_window):
        """Handle logout"""
        dashboard_window.destroy()
        self.root.deiconify()
        self.face_system.current_user = None
        self.username_entry.delete(0, tk.END)
        self.password_entry.delete(0, tk.END)
        self.status_label.config(text="Logged out successfully", fg=SUCCESS_COLOR)

    def exit_app(self):
        """Exit the application"""
        self.face_system.stop_recognition()
        self.root.destroy()

    def _darken_color(self, color, factor=0.8):
        """Darken a color by a factor"""
        r = int(int(color[1:3], 16) * factor)
        g = int(int(color[3:5], 16) * factor)
        b = int(int(color[5:7], 16) * factor)
        return f"#{r:02x}{g:02x}{b:02x}"


def main():
    root = tk.Tk()
    app = EnhancedLoginApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
