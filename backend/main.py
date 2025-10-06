from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import cv2
import mediapipe as mp
import numpy as np
import joblib
import base64
from io import BytesIO
from PIL import Image

app = FastAPI(title="StrikeSense Backend", version="1.0")

# Load trained model
model = joblib.load("ml/models/punch_classifier.pkl")

# Initialize Mediapipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

@app.get("/")
def root():
    return {"message": "Welcome to StrikeSense API"}

# ----------- Predict Punch from Uploaded Image ------------
@app.post("/predict")
async def predict_punch(file: UploadFile = File(...)):
    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if not results.pose_landmarks:
        return JSONResponse({"error": "No pose detected"}, status_code=400)

    lm = results.pose_landmarks.landmark
    key_indices = [
        mp_pose.PoseLandmark.LEFT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_ELBOW,
        mp_pose.PoseLandmark.LEFT_WRIST,
        mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.RIGHT_ELBOW,
        mp_pose.PoseLandmark.RIGHT_WRIST
    ]

    features = []
    for idx in key_indices:
        pt = lm[idx]
        features.extend([pt.x, pt.y, pt.z])

    pred = model.predict([features])[0]
    return {"prediction": pred}


# ----------- Debug Mode: Live Webcam Testing ------------
def run_live_demo():
    cap = cv2.VideoCapture(0)
    mp_drawing = mp.solutions.drawing_utils

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            key_indices = [
                mp_pose.PoseLandmark.LEFT_SHOULDER,
                mp_pose.PoseLandmark.LEFT_ELBOW,
                mp_pose.PoseLandmark.LEFT_WRIST,
                mp_pose.PoseLandmark.RIGHT_SHOULDER,
                mp_pose.PoseLandmark.RIGHT_ELBOW,
                mp_pose.PoseLandmark.RIGHT_WRIST
            ]
            features = []
            for idx in key_indices:
                pt = lm[idx]
                features.extend([pt.x, pt.y, pt.z])

            pred = model.predict([features])[0]
            cv2.putText(frame, f"Prediction: {pred}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow("StrikeSense Live", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="Run live webcam demo")
    args = parser.parse_args()

    if args.demo:
        run_live_demo()
    else:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)

