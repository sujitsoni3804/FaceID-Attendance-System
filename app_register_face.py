import os
import cv2
import pickle
import numpy as np
import mediapipe as mp
import face_recognition   # <--- CORRECT LIBRARY!

def get_face_alignment(face_landmarks, image_shape):
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

def collect_face_encodings(frames_needed=5):
    encodings = []
    cap = cv2.VideoCapture(0)
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        min_detection_confidence=0.75
    )

    print("====== Face Registration ======")
    print("Align your face to the camera. Full face must be visible and centered.")
    print(f"Capturing {frames_needed} valid face frames. Press 'q' to quit at any time.")

    collected = 0
    retry_count = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            retry_count += 1
            msg = "ERROR: Unable to read from camera. Check camera connection."
            print(msg)
            key = cv2.waitKey(1000) & 0xFF
            if key == ord('q'):
                print("Registration cancelled by user.")
                break
            continue

        display_frame = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(img_rgb)

        valid_for_encoding = False
        message = "No Face Detected - Please face the camera"

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                if get_face_alignment(face_landmarks, display_frame.shape):
                    message = f"Full Face Detected [{collected+1}/{frames_needed}]"
                    valid_for_encoding = True
                    mp.solutions.drawing_utils.draw_landmarks(
                        display_frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS,
                        mp.solutions.drawing_utils.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1)
                    )
                else:
                    message = "Face Not Fully Visible - Center and face camera"

        cv2.putText(display_frame, message, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0) if valid_for_encoding else (0,0,255), 2)

        cv2.imshow("Face Registration", display_frame)
        key = cv2.waitKey(1) & 0xFF

        # If valid face, try to extract encoding
        if valid_for_encoding:
            try:
                encoding_arr = face_recognition.face_encodings(frame)
                if len(encoding_arr) > 0:
                    encodings.append(encoding_arr[0])
                    collected += 1
                    print(f"Frame {collected} captured successfully.")
                    cv2.waitKey(500) # small pause to allow user to reposition
                else:
                    print("Could not extract encoding, try again.")
            except Exception as e:
                print(f"Error extracting face encoding: {e}")

        if collected >= frames_needed:
            print("Face frames collected successfully.")
            break

        if key == ord('q'):
            print("Registration cancelled by user.")
            break

    cap.release()
    cv2.destroyAllWindows()
    if collected == frames_needed:
        return encodings
    else:
        print("Face registration was incomplete.")
        return []

def save_encoding(name, encodings, db_dir='./db'):
    if not os.path.exists(db_dir):
        os.makedirs(db_dir)
    encoding_array = np.array(encodings)
    avg_encoding = np.mean(encoding_array, axis=0)
    with open(os.path.join(db_dir, f"{name}.pickle"), 'wb') as f:
        pickle.dump(avg_encoding, f)
    print(f"Registered {name} successfully to the DB.")

def main():
    print("Welcome to Face Registration Utility")
    print("First, capture the face data.")
    encodings = collect_face_encodings(frames_needed=5)
    if len(encodings) > 0:
        name = input("Enter the name of the user to register: ").strip()
        if name:
            save_encoding(name, encodings)
            print("Registration complete!")
        else:
            print("No name entered, registration aborted.")
    else:
        print("No valid face frames collected, registration failed.")

if __name__ == "__main__":
    main()