import cv2
import streamlit as st
from core.face_tracker import FaceTracker
from core.pose_estimator import PoseEstimator

# UI
st.set_page_config(page_title="Eyes On You", layout="wide")
st.title("Smart Exam Proctoring System")
st.write("Face Presence & Head Pose Tracking")

# Sidebar
st.sidebar.header("Controls")
run = st.sidebar.checkbox('Start Proctoring Session')

# placeholder for video and alerts
alert_placeholder = st.empty()
frame_window = st.image([])

# initialize ai models
tracker = FaceTracker()
pose = PoseEstimator()

# Main application loop
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

    # Alert logic
    if face_count == 0:
        cv2.putText(frame_rgb, "ALERT: NO FACE DETECTED", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        alert_placeholder.error("Violation: Candidate missing from frame!")
    elif face_count > 1:
        cv2.putText(frame_rgb, f"ALERT: MULTIPLE FACES ({face_count})", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        alert_placeholder.error(f"Violation: Multiple people detected ({face_count})!")
    elif looking_away:
        cv2.putText(frame_rgb, "WARNING: LOOKING AWAY", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 165, 0), 3)
        alert_placeholder.warning("Warning: Candidate looking away from screen!")
    else:
        cv2.putText(frame_rgb, "Status: Monitering...", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        alert_placeholder.success("Status: Normal")

    # pushing processed frames to streamlit ui
    frame_window.image(frame_rgb)

if not run:
    st.info("Click 'Start Proctoring Session' in the sidebar to begin.")
    camera.release()