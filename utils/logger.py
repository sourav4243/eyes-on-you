import os
import cv2
import csv
from datetime import datetime
import time

class SessionLogger:
    def __init__(self):
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = "data/logs"
        self.snap_dir = "data/snapshots"

        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.snap_dir, exist_ok=True)

        self.log_file = os.path.join(self.log_dir, f"session_{self.session_id}.csv")

        with open(self.log_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Timestamp", "Violation Type", "Evidence File"])

        # waiting 5 seconds before logging same violation again
        self.cooldowns = {
            "NO_FACE": 0,
            "MULTIPLE_FACES": 0,
            "LOOKING_AWAY": 0, 
            "UNAUTHORIZED_OBJECT": 0
        }
        self.cooldown_time = 5.0    # seconds

    def log_event(self, event_type, frame_rgb):
        current_time = time.time()

        if current_time - self.cooldowns[event_type] < self.cooldown_time:
            return
        
        self.cooldowns[event_type] = current_time

        # save snapshot
        timestamp_str = datetime.now().strftime("%H-%M-%S")
        filename = f"{event_type}_{timestamp_str}.jpg"
        filepath = os.path.join(self.snap_dir, filename)

        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filepath, frame_bgr)

        # log to csv
        human_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([human_time, event_type, filename])