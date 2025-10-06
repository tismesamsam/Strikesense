import cv2
import joblib
import mediapipe as mp
import numpy as np

# -----------------------------
# Load trained ML model
# -----------------------------
model = joblib.load("models/punch_classifier.pkl")

# -----------------------------
# Setup Mediapipe Pose
# -----------------------------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# -----------------------------
# Start webcam
# -----------------------------
cap = cv2.VideoCapture(0)

# Punch counters
jab_count, cross_count, hook_count, uppercut_count = 0, 0, 0, 0

# Key landmarks to track (shoulders, elbows, wrists)
key_indices = [
    mp_pose.PoseLandmark.LEFT_SHOULDER,
    mp_pose.PoseLandmark.LEFT_ELBOW,
    mp_pose.PoseLandmark.LEFT_WRIST,
    mp_pose.PoseLandmark.RIGHT_SHOULDER,
    mp_pose.PoseLandmark.RIGHT_ELBOW,
    mp_pose.PoseLandmark.RIGHT_WRIST
]

# -----------------------------
# Real-time loop
# -----------------------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror view
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        features = []

        # Extract selected keypoints
        for idx in key_indices:
            pt = lm[idx]
            features.extend([pt.x, pt.y, pt.z])

        # Ensure correct feature length before predicting
        if len(features) == len(key_indices) * 3:
            pred = model.predict([features])[0]

            # Increment counts based on prediction
            if pred == "jab":
                jab_count += 1
            elif pred == "cross":
                cross_count += 1
            elif pred == "hook":
                hook_count += 1
            elif pred == "uppercut":
                uppercut_count += 1

            # Display prediction
            cv2.putText(frame, f"Prediction: {pred}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        # Display counters
        cv2.putText(frame,
                    f"Jabs: {jab_count}  Crosses: {cross_count}  Hooks: {hook_count}  Uppercuts: {uppercut_count}",
                    (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Draw landmarks
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Show video
    cv2.imshow("StrikeSense ML", frame)

    # Quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -----------------------------
# Cleanup
# -----------------------------
cap.release()
cv2.destroyAllWindows()
