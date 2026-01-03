import os
import cv2
import pickle
import numpy as np
import mediapipe as mp
import face_recognition
import time
import csv
from datetime import datetime
import math

class FaceRecognitionTest:
    def __init__(self):
        # Set camera and screen resolution
        self.screen_w, self.screen_h = 1440, 900
        self.cam_w, self.cam_h = 640, 480

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam_w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_h)

        # --- CAMERA CHECK ADDED HERE ---
        if not self.cap.isOpened():
            print("ERROR: Camera could not be opened. Please check the connection.")
            exit(1)
        # --- END CAMERA CHECK ---

        self.db_dir = './db'
        self.detection_dir = './Detection_images'
        self.csv_file = './attendance_log.csv'

        if not os.path.exists(self.db_dir):
            os.mkdir(self.db_dir)
        if not os.path.exists(self.detection_dir):
            os.mkdir(self.detection_dir)

        self.init_csv_file()

        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.75
        )

        self.last_recognition_time = 0
        self.recognition_interval = 1.0
        self.current_name = None
        self.detection_start_time = 0
        self.detection_duration = 5.0
        self.image_saved = False
        self.is_paused = False
        self.pause_frame = None
        self.detection_timestamp = None
        self.current_action = None
        self.error_message = None

        self.mode = None
        self.animation_frame = 0
        self.show_animation = False

        self.animation_alpha = 0.0
        self.animation_scale = 1.0
        self.animation_progress = 0.0

        print("Face Recognition System Started")
        print("Press '1' for Check-In mode")
        print("Press '2' for Check-Out mode")
        print("Press 'q' to quit")

    def init_csv_file(self):
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Name', 'Date', 'Check-In Time', 'Check-Out Time', 'Status'])

    def get_face_alignment(self, face_landmarks, image_shape):
        ih, iw = image_shape[:2]
        LEFT_EYE, RIGHT_EYE, NOSE_TIP, MOUTH_TOP, MOUTH_BOTTOM = 33, 263, 1, 13, 14

        left_eye = face_landmarks.landmark[LEFT_EYE]
        right_eye = face_landmarks.landmark[RIGHT_EYE]
        nose_tip = face_landmarks.landmark[NOSE_TIP]
        mouth_top = face_landmarks.landmark[MOUTH_TOP]
        mouth_bottom = face_landmarks.landmark[MOUTH_BOTTOM]

        left_eye_xy = np.array([left_eye.x * iw, left_eye.y * ih])
        right_eye_xy = np.array([right_eye.x * iw, right_eye.y * ih])
        nose_xy = np.array([nose_tip.x * iw, nose_tip.y * ih])
        mouth_top_xy = np.array([mouth_top.x * iw, mouth_top.y * ih])
        mouth_bottom_xy = np.array([mouth_bottom.x * iw, mouth_bottom.y * ih])

        eye_y_diff = abs(left_eye_xy[1] - right_eye_xy[1])
        eyes_centered = eye_y_diff < 0.07 * ih

        eye_x_center = (left_eye_xy[0] + right_eye_xy[0]) / 2
        nose_x_diff = abs(nose_xy[0] - eye_x_center)
        nose_centered = nose_x_diff < 0.08 * iw

        mouth_vdist = abs(mouth_top_xy[1] - mouth_bottom_xy[1])
        mouth_visible = mouth_vdist > 0.01 * ih

        eye_x_dist = abs(left_eye_xy[0] - right_eye_xy[0])
        eyes_apart = eye_x_dist > 0.15 * iw

        side_face = (abs(nose_xy[0] - left_eye_xy[0]) < 0.04 * iw) or (abs(nose_xy[0] - right_eye_xy[0]) < 0.04 * iw)
        landmarks_count = len(face_landmarks.landmark) >= 468

        return (eyes_centered and nose_centered and mouth_visible and eyes_apart and not side_face and landmarks_count)

    def recognize_face(self, frame):
        try:
            encodings = face_recognition.face_encodings(frame)
            if len(encodings) == 0:
                return 'no_persons_found'

            encoding = encodings[0]

            for file_name in os.listdir(self.db_dir):
                if file_name.endswith('.pickle'):
                    name = file_name.split('.')[0]
                    with open(os.path.join(self.db_dir, file_name), 'rb') as f:
                        saved_encoding = pickle.load(f)
                    if face_recognition.compare_faces([saved_encoding], encoding, tolerance=0.6)[0]:
                        return name
            return 'unknown_person'
        except:
            return 'no_persons_found'

    def get_face_bounding_box(self, frame):
        try:
            face_locations = face_recognition.face_locations(frame)
            if len(face_locations) > 0:
                top, right, bottom, left = face_locations[0]
                return (left, top, right, bottom)
            return None
        except:
            return None

    def save_detection_image(self, frame, name, action):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{action}_{timestamp}.jpg"
        filepath = os.path.join(self.detection_dir, filename)
        cv2.imwrite(filepath, frame)
        print(f"Detection image saved: {filepath}")

    def get_formatted_time(self):
        now = datetime.now()
        return now.strftime("%I:%M %p")

    def get_current_date(self):
        now = datetime.now()
        return now.strftime("%Y-%m-%d")

    def get_latest_user_action(self, name):
        """Return the latest action for a user (checkin/checkout) for today, or None if not found"""
        date = self.get_current_date()
        latest_row = None
        if os.path.exists(self.csv_file):
            with open(self.csv_file, 'r', newline='') as file:
                reader = csv.reader(file)
                for row in reader:
                    if len(row) < 5:
                        continue
                    if row[0] == name and row[1] == date:
                        latest_row = row
        if latest_row:
            if latest_row[4].lower() == "checked in":
                return "checkin"
            elif latest_row[4].lower() == "checked out":
                return "checkout"
        return None

    def validate_action(self, name, requested_action):
        """
        Only allow:
        - check-in if user is new (no logs today) or latest is checkout
        - check-out if latest is checkin
        """
        latest_action = self.get_latest_user_action(name)
        print(f"DEBUG: User {name} latest action today: {latest_action}, requested action: {requested_action}")

        if requested_action == "checkin":
            if latest_action is None or latest_action == "checkout":
                return True, "Check-in allowed"
            elif latest_action == "checkin":
                return False, "Already checked in. Please check out first."
        elif requested_action == "checkout":
            if latest_action == "checkin":
                return True, "Check-out allowed"
            elif latest_action is None or latest_action == "checkout":
                return False, "Check-in required before check-out."
        return False, "Invalid action"

    def log_to_csv(self, name, action, timestamp):
        date = self.get_current_date()
        if action == "checkin":
            row = [name, date, timestamp, "", "Checked In"]
        elif action == "checkout":
            row = [name, date, "", timestamp, "Checked Out"]
        else:
            return
        with open(self.csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(row)

    def draw_professional_animation(self, frame):
        """Draw professional success animation adapted for low-res camera and screen"""
        if self.show_animation:
            h, w = frame.shape[:2]
            # Adapt container size and positions for smaller camera frames
            center_x, center_y = w // 2, h // 2

            elapsed_time = cv2.getTickCount() / cv2.getTickFrequency() - self.detection_start_time
            self.animation_progress = min(elapsed_time / self.detection_duration, 1.0)

            overlay = frame.copy()
            # Fade in/out
            if self.animation_progress < 0.2:
                self.animation_alpha = self.animation_progress * 5
            elif self.animation_progress > 0.8:
                self.animation_alpha = (1.0 - self.animation_progress) * 5
            else:
                self.animation_alpha = 1.0

            # Draw semi-transparent background
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
            cv2.addWeighted(frame, 0.3, overlay, 0.7 * self.animation_alpha, 0, frame)

            scale_factor = 0.7 + 0.25 * math.sin(self.animation_progress * math.pi)

            # Container - scaled for camera frame
            container_w, container_h = int(w * 0.8), int(h * 0.5)
            container_x = center_x - container_w // 2
            container_y = center_y - container_h // 2

            cv2.rectangle(frame, (container_x, container_y),
                          (container_x + container_w, container_y + container_h),
                          (40, 40, 40), -1)

            border_color = (0, 255, 0) if self.current_action == "checkin" else (0, 165, 255)
            cv2.rectangle(frame, (container_x, container_y),
                          (container_x + container_w, container_y + container_h),
                          border_color, 4)

            # Success Icon
            icon_size = int(36 * scale_factor)
            icon_x = container_x + int(container_w * 0.13)
            icon_y = container_y + container_h // 2
            cv2.circle(frame, (icon_x, icon_y), icon_size, border_color, -1)
            cv2.circle(frame, (icon_x, icon_y), icon_size, (255, 255, 255), 2)

            check_size = int(icon_size * 0.7)
            cv2.line(frame, (icon_x - check_size // 2, icon_y),
                     (icon_x - check_size // 6, icon_y + check_size // 2), (255, 255, 255), 4)
            cv2.line(frame, (icon_x - check_size // 6, icon_y + check_size // 2),
                     (icon_x + check_size // 2, icon_y - check_size // 2), (255, 255, 255), 4)

            # Text positions scaled for small frame
            text_x = container_x + int(container_w * 0.30)
            text_y = container_y + int(container_h * 0.23)

            title_text = f"{self.current_action.upper()} SUCCESSFUL"
            cv2.putText(frame, title_text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

            name_text = f"User: {self.current_name.title()}"
            cv2.putText(frame, name_text, (text_x, text_y + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, border_color, 2)

            time_text = f"Time: {self.detection_timestamp}"
            cv2.putText(frame, time_text, (text_x, text_y + 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

            status_text = f"{self.current_action.title()} completed successfully"
            cv2.putText(frame, status_text, (text_x, text_y + 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)

            # Progress bar scaled for frame
            progress_w = int(container_w * 0.7)
            progress_h = 6
            progress_x = center_x - progress_w // 2
            progress_y = container_y + container_h - 28

            cv2.rectangle(frame, (progress_x, progress_y),
                          (progress_x + progress_w, progress_y + progress_h),
                          (80, 80, 80), -1)

            filled_w = int(progress_w * self.animation_progress)
            cv2.rectangle(frame, (progress_x, progress_y),
                          (progress_x + filled_w, progress_y + progress_h),
                          border_color, -1)

            # Animated particles effect scaled down
            for i in range(7):
                angle = (self.animation_progress * 360 + i * 50) % 360
                particle_x = int(center_x + int(container_w * 0.20) * math.cos(math.radians(angle)))
                particle_y = int(center_y + int(container_h * 0.40) * math.sin(math.radians(angle)))
                particle_size = int(4 + 2 * math.sin(self.animation_progress * math.pi * 2 + i))
                cv2.circle(frame, (particle_x, particle_y), particle_size, border_color, -1)

    def draw_error_message(self, frame, name, bbox, timestamp, error_msg):
        h, w = frame.shape[:2]
        center_x, center_y = w // 2, h // 2

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(frame, 0.3, overlay, 0.7, 0, frame)

        container_w, container_h = int(w * 0.7), int(h * 0.35)
        container_x = center_x - container_w // 2
        container_y = center_y - container_h // 2

        cv2.rectangle(frame, (container_x, container_y),
                      (container_x + container_w, container_y + container_h),
                      (40, 40, 40), -1)

        cv2.rectangle(frame, (container_x, container_y),
                      (container_x + container_w, container_y + container_h),
                      (0, 0, 255), 4)

        icon_size = 32
        icon_x = container_x + int(container_w * 0.13)
        icon_y = container_y + container_h // 2

        cv2.circle(frame, (icon_x, icon_y), icon_size, (0, 0, 255), -1)
        cv2.circle(frame, (icon_x, icon_y), icon_size, (255, 255, 255), 2)

        cv2.line(frame, (icon_x - 18, icon_y - 18), (icon_x + 18, icon_y + 18), (255, 255, 255), 4)
        cv2.line(frame, (icon_x + 18, icon_y - 18), (icon_x - 18, icon_y + 18), (255, 255, 255), 4)

        text_x = container_x + int(container_w * 0.27)
        text_y = container_y + int(container_h * 0.25)

        cv2.putText(frame, "ERROR", (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        cv2.putText(frame, f"User: {name.title()}", (text_x, text_y + 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.putText(frame, error_msg, (text_x, text_y + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        cv2.putText(frame, f"Time: {timestamp}", (text_x, text_y + 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    def draw_mode_indicator(self, frame):
        h, w = frame.shape[:2]
        if self.mode == "checkin":
            cv2.rectangle(frame, (10, h - 54), (int(w * 0.36), h - 10), (0, 255, 0), -1)
            cv2.putText(frame, "MODE: CHECK-IN", (18, h - 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        elif self.mode == "checkout":
            cv2.rectangle(frame, (10, h - 54), (int(w * 0.40), h - 10), (0, 165, 255), -1)
            cv2.putText(frame, "MODE: CHECK-OUT", (18, h - 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        else:
            cv2.rectangle(frame, (10, h - 54), (int(w * 0.60), h - 10), (100, 100, 100), -1)
            cv2.putText(frame, "Press '1' for Check-In or '2' for Check-Out", (18, h - 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    def run(self):
        print("Starting face recognition system...")

        window_name = "Face Recognition System"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        # Make window full-screen
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                print("ERROR: Unable to read frame from camera. Terminating application.")
                break

            display_frame = cv2.resize(frame, (self.screen_w, self.screen_h))

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('1'):
                self.mode = "checkin"
                print("Switched to Check-In mode")
            elif key == ord('2'):
                self.mode = "checkout"
                print("Switched to Check-Out mode")
            elif key == 27:
                break

            current_time = cv2.getTickCount() / cv2.getTickFrequency()

            if self.is_paused:
                pause_disp = display_frame.copy()
                remaining_time = self.detection_duration - (current_time - self.detection_start_time)
                if remaining_time > 0:
                    if self.error_message:
                        bbox = self.get_face_bounding_box(display_frame)
                        self.draw_error_message(pause_disp, self.current_name, bbox,
                                               self.detection_timestamp, self.error_message)
                    else:
                        self.draw_professional_animation(pause_disp)
                else:
                    self.is_paused = False
                    self.current_name = None
                    self.pause_frame = None
                    self.detection_timestamp = None
                    self.current_action = None
                    self.error_message = None
                    self.show_animation = False
                    self.animation_frame = 0
                    self.animation_progress = 0.0
                    print("Resuming normal operation...")

                self.draw_mode_indicator(pause_disp)
                cv2.imshow(window_name, pause_disp)
            else:
                if self.mode:
                    img_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                    results = self.face_mesh.process(img_rgb)
                    if results.multi_face_landmarks:
                        for face_landmarks in results.multi_face_landmarks:
                            if self.get_face_alignment(face_landmarks, display_frame.shape):
                                if current_time - self.last_recognition_time > self.recognition_interval:
                                    new_name = self.recognize_face(frame)
                                    if new_name and new_name not in ['no_persons_found', 'unknown_person']:
                                        is_valid, message = self.validate_action(new_name, self.mode)
                                        self.current_name = new_name
                                        self.current_action = self.mode
                                        self.detection_start_time = current_time
                                        self.detection_timestamp = self.get_formatted_time()
                                        self.pause_frame = display_frame.copy()
                                        self.is_paused = True
                                        if is_valid:
                                            self.error_message = None
                                            self.show_animation = True
                                            self.save_detection_image(frame, self.current_name, self.mode)
                                            self.log_to_csv(self.current_name, self.mode, self.detection_timestamp)
                                            print(f"SUCCESS: {self.current_name} {self.mode}ed at {self.detection_timestamp}")
                                        else:
                                            self.error_message = message
                                            self.show_animation = False
                                            print(f"ERROR: {self.current_name} - {message}")
                                    elif new_name == 'unknown_person':
                                        self.current_name = new_name
                                        self.current_action = None
                                        self.detection_start_time = current_time
                                        self.detection_timestamp = self.get_formatted_time()
                                        self.pause_frame = display_frame.copy()
                                        self.is_paused = True
                                        self.error_message = "Unknown person detected"
                                        self.show_animation = False
                                        print(f"Unknown person detected at {self.detection_timestamp}")
                                    self.last_recognition_time = current_time
                                cv2.putText(display_frame, "Full Face Detected", (30, 50),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                            else:
                                cv2.putText(display_frame, "Face Not Fully Visible", (30, 50),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                    else:
                        cv2.putText(display_frame, "No Face Detected", (30, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                self.draw_mode_indicator(display_frame)
                cv2.imshow(window_name, display_frame)

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = FaceRecognitionTest()
    app.run()