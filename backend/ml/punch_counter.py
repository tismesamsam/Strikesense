# punch_counter.py
import cv2
import mediapipe as mp
import threading

class PunchCounter:
    def __init__(self):
        self.count = 0
        self.running = True
        # start punch detection in a background thread
        t = threading.Thread(target=self.run, daemon=True)
        t.start()

    def run(self):
        cap = cv2.VideoCapture(0)
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
            # 👊 your punch detection logic here
            # if punch detected:
            #     self.count += 1
        cap.release()

    def get_count(self):
        return self.count
