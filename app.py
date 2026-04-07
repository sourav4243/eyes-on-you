import cv2
import streamlit as st
import pandas as pd
import os
from core.face_tracker import FaceTracker
from core.pose_estimator import PoseEstimator
from core.object_detector import ObjectDetector
from utils.logger import SessionLogger

# UI
st.set_page_config(page_title="Eyes On You", layout="wide")
st.title("Smart Exam Proctoring System")
st.write("Face Presence & Head Pose Tracking")

# Sidebar
st.sidebar.header("Controls")
run = st.sidebar.checkbox('Start Proctoring Session')
view_report = st.sidebar.checkbox('View Session Report')


if 'logger' not in st.session_state:
    st.session_state.logger = SessionLogger()

# initialize ai models
tracker = FaceTracker()
pose = PoseEstimator()
detector = ObjectDetector()

if run and not view_report:
    alert_placeholder = st.empty()
    frame_window = st.image([])
    camera = cv2.VideoCapture(0)
    

    while run: 
        success, frame = camera.read()
        if not success:
            st.error("Failed to access the webcam.")
            break

        # opencv uses BGR, but mediapipe and streamlit need RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # process frame through both models
        face_count = tracker.process(frame_rgb)
        looking_away = pose.process(frame_rgb, face_count)
        object_detected = detector.process(frame_rgb)

        # Alert logic
        if face_count == 0:
            cv2.putText(frame_rgb, "ALERT: NO FACE DETECTED", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
            alert_placeholder.error("Violation: Candidate missing from frame!")
            st.session_state.logger.log_event("NO_FACE", frame_rgb)

        elif face_count > 1:
            cv2.putText(frame_rgb, f"ALERT: MULTIPLE FACES ({face_count})", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
            alert_placeholder.error(f"Violation: Multiple people detected ({face_count})!")
            st.session_state.logger.log_event("MULTIPLE_FACES", frame_rgb)

        elif object_detected: 
            cv2.putText(frame_rgb, "ALERT: PHONE/BOOK DETECTED", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
            alert_placeholder.error("Violation: Unauthorized object detected!")
            st.session_state.logger.log_event("UNAUTHORIZED_OBJECT", frame_rgb)

        elif looking_away:
            cv2.putText(frame_rgb, "WARNING: LOOKING AWAY", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 165, 0), 3)
            alert_placeholder.warning("Warning: Candidate looking away from screen!")
            st.session_state.logger.log_event("LOOKING_AWAY", frame_rgb)

        else:
            cv2.putText(frame_rgb, "Status: Monitering...", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            alert_placeholder.success("Status: Normal")

        # pushing processed frames to streamlit ui
        frame_window.image(frame_rgb)
    
    camera.release()

elif view_report:
    st.subheader("Session Incident Report")
    st.info("Proctoring paused. Viewing logs")

    log_file = st.session_state.logger.log_file
    if os.path.exists(log_file):
        df = pd.read_csv(log_file)
        if not df.empty:
            st.dataframe(df, use_container_width=True)
            st.write("Evidence Snapshots are saved in `data/snapshots/`")
        else:
            st.success("No violations recorded in this session!")
    else:
        st.write("No log file found for this sessoin yet.")

else:
    st.info("Click 'Start Proctoring Session' in the sidebar to begin.")