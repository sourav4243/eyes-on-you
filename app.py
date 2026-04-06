import cv2
import streamlit as st
import mediapipe as mp

# UI
st.set_page_config(page_title="Eyes On You", layout="wide")
st.title("Smart Exam Proctoring System")
st.write("Face Presence & Counting")

# Sidebar
st.sidebar.header("Controls")
run = st.sidebar.checkbox('Start Proctoring Session')

# placeholder for video and alerts
alert_placeholder = st.empty()
frame_window = st.image([])

# AI MODEL SETUP
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Main application loop
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

camera = cv2.VideoCapture(0)

while run: 
    success, frame = camera.read()
    if not success:
        st.error("Failed to access the webcam.")
        break

    # opencv uses BGR, but mediapipe and streamlit need RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_detection.process(frame_rgb)

    face_count = 0

    if results.detections:
        face_count = len(results.detections)
        for detection in results.detections:
            mp_drawing.draw_detection(frame_rgb, detection)


    if face_count == 0:
        cv2.putText(frame_rgb, "ALERT: NO FACE DETECTED", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        alert_placeholder.error("Violation: Candidate missing from frame!")
    elif face_count > 1:
        cv2.putText(frame_rgb, f"ALERT: MULTIPLE FACES ({face_count})", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        alert_placeholder.error(f"Violation: Multiple people detected ({face_count})!")
    else:
        cv2.putText(frame_rgb, "Status: Monitering...", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        alert_placeholder.success("Status: Normal")

    # pushing processed frames to streamlit ui
    frame_window.image(frame_rgb)

else:
    st.info("Click 'Start Proctoring Session' in the sidebar to begin.")
    camera.release()