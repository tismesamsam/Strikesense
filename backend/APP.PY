from flask import Flask, jsonify
import cv2
import mediapipe as mp

app = Flask(__name__)


mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
cap = cv2.VideoCapture(0)

punch_count = 0
prev_right_x = None
prev_left_x = None

def detect_punches():
    global punch_count, prev_right_x, prev_left_x
    ret, frame = cap.read()
    if not ret:
        return punch_count
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)
    
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]

        # Right hand punch
        if prev_right_x is not None and right_wrist.x - prev_right_x > 0.1:
            punch_count += 1
        prev_right_x = right_wrist.x

        # Left hand punch
        if prev_left_x is not None and left_wrist.x - prev_left_x > 0.1:
            punch_count += 1
        prev_left_x = left_wrist.x

    return punch_count

@app.route("/punches", methods=["GET"])

def get_punches():
    count = detect_punches()
    return jsonify({"punches": count})


if __name__ == "__main__":
    app.run(debug=True)  
