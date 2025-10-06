import cv2
import mediapipe as mp
import pandas as pd
import os

# Setup
DATASET_DIR = "dataset"
os.makedirs(DATASET_DIR, exist_ok=True)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

PUNCHES = ["jab", "cross", "hook", "uppercut"]

def collect(punch_type, samples=20):
    cap = cv2.VideoCapture(2)
    count = 0
    print(f"🎥 Collecting {samples} samples for: {punch_type}")
    print("👉 Press SPACEBAR when you throw a punch. Press Q to quit.")

    while count < samples:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            # Draw skeleton
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.putText(frame, f"{punch_type}: {count}/{samples}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Collect Data", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == 32 and results.pose_landmarks:  # SPACEBAR = 32
            # Save landmarks
            lm = results.pose_landmarks.landmark
            key_indices = [
                mp_pose.PoseLandmark.LEFT_SHOULDER,
                mp_pose.PoseLandmark.LEFT_ELBOW,
                mp_pose.PoseLandmark.LEFT_WRIST,
                mp_pose.PoseLandmark.RIGHT_SHOULDER,
                mp_pose.PoseLandmark.RIGHT_ELBOW,
                mp_pose.PoseLandmark.RIGHT_WRIST
            ]

            row = []
            for idx in key_indices:
                pt = lm[idx]
                row.extend([pt.x, pt.y, pt.z])

            df = pd.DataFrame([row])
            filename = os.path.join(DATASET_DIR, f"{punch_type}_{count}.csv")
            df.to_csv(filename, index=False)

            count += 1
            print(f"✅ Saved {punch_type}_{count}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"🎉 Done collecting {punch_type} data.")

if __name__ == "__main__":
    for punch in PUNCHES:
        collect(punch, samples=20)
