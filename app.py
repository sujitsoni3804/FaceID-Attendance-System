import os
import cv2
import pickle
import numpy as np
import mediapipe as mp
import face_recognition
import csv
from datetime import datetime
import math

class FaceAttendanceSystem:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("ERROR: Camera not accessible. Check connection.")
            exit(1)
        
        self.cam_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.cam_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.screen_w, self.screen_h = self.get_screen_resolution()
        
        print(f"Camera: {self.cam_w}x{self.cam_h} | Display: {self.screen_w}x{self.screen_h}")
        
        self.db_dir = './db'
        self.detection_dir = './Detection_images'
        self.csv_file = './attendance_log.csv'
        
        os.makedirs(self.db_dir, exist_ok=True)
        os.makedirs(self.detection_dir, exist_ok=True)
        self.init_csv()
        
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.75
        )
        
        self.reset_state()
    
    def get_screen_resolution(self):
        try:
            import tkinter as tk
            root = tk.Tk()
            root.withdraw()
            return root.winfo_screenwidth(), root.winfo_screenheight()
        except:
            return 1920, 1080
    
    def reset_state(self):
        self.mode = None
        self.current_name = None
        self.is_paused = False
        self.detection_start_time = 0
        self.detection_duration = 5.0
        self.cooldown_time = 3.0
        self.last_detection_time = 0
        self.detection_timestamp = None
        self.current_action = None
        self.error_message = None
        self.show_animation = False
        self.animation_progress = 0.0
    
    def init_csv(self):
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, 'w', newline='') as f:
                csv.writer(f).writerow(['Name', 'Date', 'Check-In Time', 'Check-Out Time', 'Status'])
    
    def scale_pos(self, x, y, ref_w=None, ref_h=None):
        ref_w = ref_w or self.screen_w
        ref_h = ref_h or self.screen_h
        return int(x * ref_w), int(y * ref_h)
    
    def scale_val(self, val, is_width=True, ref_w=None, ref_h=None):
        ref_w = ref_w or self.screen_w
        ref_h = ref_h or self.screen_h
        return int(val * (ref_w if is_width else ref_h))
    
    def get_face_alignment(self, face_landmarks, img_shape):
        ih, iw = img_shape[:2]
        LEFT_EYE, RIGHT_EYE, NOSE_TIP, MOUTH_TOP, MOUTH_BOTTOM = 33, 263, 1, 13, 14
        
        landmarks = {
            'left_eye': np.array([face_landmarks.landmark[LEFT_EYE].x * iw, face_landmarks.landmark[LEFT_EYE].y * ih]),
            'right_eye': np.array([face_landmarks.landmark[RIGHT_EYE].x * iw, face_landmarks.landmark[RIGHT_EYE].y * ih]),
            'nose': np.array([face_landmarks.landmark[NOSE_TIP].x * iw, face_landmarks.landmark[NOSE_TIP].y * ih]),
            'mouth_top': np.array([face_landmarks.landmark[MOUTH_TOP].x * iw, face_landmarks.landmark[MOUTH_TOP].y * ih]),
            'mouth_bottom': np.array([face_landmarks.landmark[MOUTH_BOTTOM].x * iw, face_landmarks.landmark[MOUTH_BOTTOM].y * ih])
        }
        
        eye_y_diff = abs(landmarks['left_eye'][1] - landmarks['right_eye'][1])
        eyes_level = eye_y_diff < 0.07 * ih
        
        eye_center_x = (landmarks['left_eye'][0] + landmarks['right_eye'][0]) / 2
        nose_centered = abs(landmarks['nose'][0] - eye_center_x) < 0.08 * iw
        
        mouth_visible = abs(landmarks['mouth_top'][1] - landmarks['mouth_bottom'][1]) > 0.01 * ih
        eyes_apart = abs(landmarks['left_eye'][0] - landmarks['right_eye'][0]) > 0.15 * iw
        
        not_profile = not ((abs(landmarks['nose'][0] - landmarks['left_eye'][0]) < 0.04 * iw) or 
                          (abs(landmarks['nose'][0] - landmarks['right_eye'][0]) < 0.04 * iw))
        
        return eyes_level and nose_centered and mouth_visible and eyes_apart and not_profile
    
    def recognize_face(self, frame):
        try:
            encodings = face_recognition.face_encodings(frame)
            if not encodings:
                return 'no_persons_found'
            
            for fname in os.listdir(self.db_dir):
                if fname.endswith('.pickle'):
                    name = fname.split('.')[0]
                    with open(os.path.join(self.db_dir, fname), 'rb') as f:
                        saved_enc = pickle.load(f)
                    if face_recognition.compare_faces([saved_enc], encodings[0], tolerance=0.6)[0]:
                        return name
            return 'unknown_person'
        except:
            return 'no_persons_found'
    
    def get_latest_action(self, name):
        date = datetime.now().strftime("%Y-%m-%d")
        if not os.path.exists(self.csv_file):
            return None
        
        with open(self.csv_file, 'r', newline='') as f:
            rows = [r for r in csv.reader(f) if len(r) >= 5 and r[0] == name and r[1] == date]
            if rows:
                return "checkin" if rows[-1][4].lower() == "checked in" else "checkout"
        return None
    
    def validate_action(self, name, action):
        latest = self.get_latest_action(name)
        if action == "checkin":
            return (latest != "checkin", "Already checked in today" if latest == "checkin" else "Check-in allowed")
        elif action == "checkout":
            return (latest == "checkin", "Must check-in first" if latest != "checkin" else "Check-out allowed")
        return False, "Invalid action"
    
    def log_attendance(self, name, action):
        date = datetime.now().strftime("%Y-%m-%d")
        time = datetime.now().strftime("%I:%M %p")
        row = [name, date, time, "", "Checked In"] if action == "checkin" else [name, date, "", time, "Checked Out"]
        with open(self.csv_file, 'a', newline='') as f:
            csv.writer(f).writerow(row)
    
    def save_detection_image(self, frame, name, action):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(os.path.join(self.detection_dir, f"{name}_{action}_{timestamp}.jpg"), frame)
    
    def draw_animation(self, frame):
        if not self.show_animation:
            return
        
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        
        elapsed = cv2.getTickCount() / cv2.getTickFrequency() - self.detection_start_time
        self.animation_progress = min(elapsed / self.detection_duration, 1.0)
        
        alpha = min(self.animation_progress * 5, 1.0) if self.animation_progress < 0.2 else \
                (1.0 - self.animation_progress) * 5 if self.animation_progress > 0.8 else 1.0
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(frame, 0.3, overlay, 0.7 * alpha, 0, frame)
        
        scale = 0.7 + 0.25 * math.sin(self.animation_progress * math.pi)
        
        cw, ch = int(w * 0.7), int(h * 0.45)
        cx1, cy1 = cx - cw // 2, cy - ch // 2
        
        cv2.rectangle(frame, (cx1, cy1), (cx1 + cw, cy1 + ch), (40, 40, 40), -1)
        
        color = (0, 255, 0) if self.current_action == "checkin" else (0, 165, 255)
        cv2.rectangle(frame, (cx1, cy1), (cx1 + cw, cy1 + ch), color, 4)
        
        icon_r = int(min(w, h) * 0.04 * scale)
        icon_x, icon_y = cx1 + int(cw * 0.15), cy1 + ch // 2
        cv2.circle(frame, (icon_x, icon_y), icon_r, color, -1)
        cv2.circle(frame, (icon_x, icon_y), icon_r, (255, 255, 255), 2)
        
        check_s = int(icon_r * 0.7)
        cv2.line(frame, (icon_x - check_s // 2, icon_y), (icon_x - check_s // 6, icon_y + check_s // 2), (255, 255, 255), 3)
        cv2.line(frame, (icon_x - check_s // 6, icon_y + check_s // 2), (icon_x + check_s // 2, icon_y - check_s // 2), (255, 255, 255), 3)
        
        tx = cx1 + int(cw * 0.32)
        ty = cy1 + int(ch * 0.25)
        
        font_scale_title = min(w, h) / 800
        font_scale_text = min(w, h) / 1000
        
        cv2.putText(frame, f"{self.current_action.upper()} SUCCESSFUL", (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale_title, (255, 255, 255), 2)
        cv2.putText(frame, f"User: {self.current_name.title()}", (tx, ty + int(ch * 0.15)), cv2.FONT_HERSHEY_SIMPLEX, font_scale_text, color, 2)
        cv2.putText(frame, f"Time: {self.detection_timestamp}", (tx, ty + int(ch * 0.30)), cv2.FONT_HERSHEY_SIMPLEX, font_scale_text, (200, 200, 200), 2)
        cv2.putText(frame, f"{self.current_action.title()} completed", (tx, ty + int(ch * 0.42)), cv2.FONT_HERSHEY_SIMPLEX, font_scale_text, (180, 180, 180), 2)
        
        pw = int(cw * 0.7)
        ph = max(4, int(h * 0.008))
        px, py = cx - pw // 2, cy1 + ch - int(ch * 0.12)
        
        cv2.rectangle(frame, (px, py), (px + pw, py + ph), (80, 80, 80), -1)
        cv2.rectangle(frame, (px, py), (px + int(pw * self.animation_progress), py + ph), color, -1)
        
        for i in range(8):
            angle = (self.animation_progress * 360 + i * 45) % 360
            px_part = int(cx + int(cw * 0.25) * math.cos(math.radians(angle)))
            py_part = int(cy + int(ch * 0.35) * math.sin(math.radians(angle)))
            particle_r = max(2, int(min(w, h) * 0.004 * (1 + 0.5 * math.sin(self.animation_progress * math.pi * 2 + i))))
            cv2.circle(frame, (px_part, py_part), particle_r, color, -1)
    
    def draw_error(self, frame, name, error_msg):
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(frame, 0.3, overlay, 0.7, 0, frame)
        
        cw, ch = int(w * 0.65), int(h * 0.35)
        cx1, cy1 = cx - cw // 2, cy - ch // 2
        
        cv2.rectangle(frame, (cx1, cy1), (cx1 + cw, cy1 + ch), (40, 40, 40), -1)
        cv2.rectangle(frame, (cx1, cy1), (cx1 + cw, cy1 + ch), (0, 0, 255), 4)
        
        icon_r = int(min(w, h) * 0.035)
        icon_x, icon_y = cx1 + int(cw * 0.15), cy1 + ch // 2
        cv2.circle(frame, (icon_x, icon_y), icon_r, (0, 0, 255), -1)
        cv2.circle(frame, (icon_x, icon_y), icon_r, (255, 255, 255), 2)
        
        offset = int(icon_r * 0.6)
        cv2.line(frame, (icon_x - offset, icon_y - offset), (icon_x + offset, icon_y + offset), (255, 255, 255), 3)
        cv2.line(frame, (icon_x + offset, icon_y - offset), (icon_x - offset, icon_y + offset), (255, 255, 255), 3)
        
        tx = cx1 + int(cw * 0.30)
        ty = cy1 + int(ch * 0.25)
        
        font_scale = min(w, h) / 1000
        
        cv2.putText(frame, "ERROR", (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 1.2, (255, 255, 255), 2)
        cv2.putText(frame, f"User: {name.title()}", (tx, ty + int(ch * 0.20)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 2)
        cv2.putText(frame, error_msg, (tx, ty + int(ch * 0.36)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)
        cv2.putText(frame, f"Time: {self.detection_timestamp}", (tx, ty + int(ch * 0.50)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (200, 200, 200), 2)
    
    def draw_mode_indicator(self, frame):
        h, w = frame.shape[:2]
        bh = int(h * 0.06)
        by = h - bh - int(h * 0.015)
        
        font_scale = min(w, h) / 1200
        
        if self.mode == "checkin":
            bw = int(w * 0.25)
            cv2.rectangle(frame, (int(w * 0.015), by), (int(w * 0.015) + bw, by + bh), (0, 255, 0), -1)
            cv2.putText(frame, "MODE: CHECK-IN", (int(w * 0.025), by + int(bh * 0.65)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2)
        elif self.mode == "checkout":
            bw = int(w * 0.28)
            cv2.rectangle(frame, (int(w * 0.015), by), (int(w * 0.015) + bw, by + bh), (0, 165, 255), -1)
            cv2.putText(frame, "MODE: CHECK-OUT", (int(w * 0.025), by + int(bh * 0.65)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2)
        else:
            bw = int(w * 0.50)
            cv2.rectangle(frame, (int(w * 0.015), by), (int(w * 0.015) + bw, by + bh), (100, 100, 100), -1)
            cv2.putText(frame, "Press '1' Check-In | '2' Check-Out | 'Q' Quit", (int(w * 0.025), by + int(bh * 0.65)), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, (255, 255, 255), 2)
    
    def run_attendance(self):
        window_name = "Attendance System"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        print("\n=== ATTENDANCE MODE ===")
        print("1: Check-In | 2: Check-Out | Q: Quit")
        
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                print("ERROR: Frame read failed")
                break
            
            display = cv2.resize(frame, (self.screen_w, self.screen_h))
            current_time = cv2.getTickCount() / cv2.getTickFrequency()
            
            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), ord('Q'), 27]:
                break
            elif key == ord('1'):
                self.mode = "checkin"
                print("Switched to Check-In")
            elif key == ord('2'):
                self.mode = "checkout"
                print("Switched to Check-Out")
            
            if self.is_paused:
                pause_display = display.copy()
                if current_time - self.detection_start_time < self.detection_duration:
                    if self.error_message:
                        self.draw_error(pause_display, self.current_name, self.error_message)
                    else:
                        self.draw_animation(pause_display)
                else:
                    self.reset_state()
                    print("Resuming...")
                
                self.draw_mode_indicator(pause_display)
                cv2.imshow(window_name, pause_display)
            else:
                if self.mode and (current_time - self.last_detection_time > self.cooldown_time):
                    img_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
                    results = self.face_mesh.process(img_rgb)
                    
                    if results.multi_face_landmarks:
                        for landmarks in results.multi_face_landmarks:
                            if self.get_face_alignment(landmarks, display.shape):
                                name = self.recognize_face(frame)
                                
                                if name not in ['no_persons_found', 'unknown_person']:
                                    is_valid, msg = self.validate_action(name, self.mode)
                                    self.current_name = name
                                    self.current_action = self.mode
                                    self.detection_start_time = current_time
                                    self.last_detection_time = current_time
                                    self.detection_timestamp = datetime.now().strftime("%I:%M %p")
                                    self.is_paused = True
                                    
                                    if is_valid:
                                        self.error_message = None
                                        self.show_animation = True
                                        self.save_detection_image(frame, name, self.mode)
                                        self.log_attendance(name, self.mode)
                                        print(f"SUCCESS: {name} {self.mode} at {self.detection_timestamp}")
                                    else:
                                        self.error_message = msg
                                        self.show_animation = False
                                        print(f"ERROR: {name} - {msg}")
                                
                                elif name == 'unknown_person':
                                    self.current_name = name
                                    self.detection_start_time = current_time
                                    self.last_detection_time = current_time
                                    self.detection_timestamp = datetime.now().strftime("%I:%M %p")
                                    self.is_paused = True
                                    self.error_message = "Unknown person"
                                    self.show_animation = False
                                    print(f"Unknown person at {self.detection_timestamp}")
                                
                                font_scale = min(self.screen_w, self.screen_h) / 1000
                                cv2.putText(display, "Full Face Detected", (int(self.screen_w * 0.03), int(self.screen_h * 0.06)), 
                                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2)
                            else:
                                font_scale = min(self.screen_w, self.screen_h) / 1000
                                cv2.putText(display, "Face Not Fully Visible", (int(self.screen_w * 0.03), int(self.screen_h * 0.06)), 
                                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 2)
                    else:
                        font_scale = min(self.screen_w, self.screen_h) / 1000
                        cv2.putText(display, "No Face Detected", (int(self.screen_w * 0.03), int(self.screen_h * 0.06)), 
                                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 2)
                
                self.draw_mode_indicator(display)
                cv2.imshow(window_name, display)
        
        cv2.destroyAllWindows()
    
    def run_registration(self):
        window_name = "Face Registration"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        frames_needed = 7
        
        while True:
            name = input("\nEnter name to register (or 'q' to quit): ").strip()
            if name.lower() == 'q':
                break
            if not name:
                print("Invalid name, try again")
                continue
            
            print(f"\n=== Registering: {name} ===")
            print(f"Align face, capture {frames_needed} frames. Press 'Q' to cancel.")
            
            encodings = []
            collected = 0
            
            while collected < frames_needed:
                ret, frame = self.cap.read()
                if not ret:
                    print("ERROR: Frame read failed")
                    break
                
                display = cv2.resize(frame, (self.screen_w, self.screen_h))
                img_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(img_rgb)
                
                valid = False
                msg = "No Face Detected"
                
                if results.multi_face_landmarks:
                    for landmarks in results.multi_face_landmarks:
                        if self.get_face_alignment(landmarks, display.shape):
                            valid = True
                            msg = f"Full Face Detected [{collected+1}/{frames_needed}]"
                            self.mp_drawing.draw_landmarks(display, landmarks, self.mp_face_mesh.FACEMESH_CONTOURS)
                        else:
                            msg = "Face Not Fully Visible"
                
                font_scale = min(self.screen_w, self.screen_h) / 800
                cv2.putText(display, msg, (int(self.screen_w * 0.03), int(self.screen_h * 0.08)), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0) if valid else (0, 0, 255), 2)
                
                cv2.imshow(window_name, display)
                key = cv2.waitKey(1) & 0xFF
                
                if valid:
                    try:
                        enc_list = face_recognition.face_encodings(frame)
                        if enc_list:
                            encodings.append(enc_list[0])
                            collected += 1
                            print(f"Frame {collected}/{frames_needed} captured")
                            cv2.waitKey(300)
                    except Exception as e:
                        print(f"Encoding error: {e}")
                
                if key in [ord('q'), ord('Q')]:
                    print("Registration cancelled")
                    encodings = []
                    break
            
            if len(encodings) == frames_needed:
                avg_enc = np.mean(np.array(encodings), axis=0)
                with open(os.path.join(self.db_dir, f"{name}.pickle"), 'wb') as f:
                    pickle.dump(avg_enc, f)
                print(f"✓ {name} registered successfully!")
            else:
                print("Registration incomplete")
        
        cv2.destroyAllWindows()
    
    def start(self):
        print("\n" + "="*50)
        print("FACE RECOGNITION ATTENDANCE SYSTEM")
        print("="*50)
        print("\nSelect Mode:")
        print("  [R] Registration Mode")
        print("  [A] Attendance Mode")
        print("  [Q] Quit")
        
        while True:
            choice = input("\nEnter choice: ").strip().upper()
            if choice == 'R':
                self.run_registration()
                print("\nRestart required. Exiting...")
                break
            elif choice == 'A':
                self.run_attendance()
                break
            elif choice == 'Q':
                break
            else:
                print("Invalid choice. Try again.")
        
        self.cap.release()
        cv2.destroyAllWindows()
        print("\nSystem terminated.")

if __name__ == "__main__":
    app = FaceAttendanceSystem()
    app.start()