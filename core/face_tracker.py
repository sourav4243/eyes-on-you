import mediapipe as mp

class FaceTracker:
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.detector = self.mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.5
        )

    def process(self, frame_rgb):
        """Processes the frame, draws boxes, and returns the face count."""
        results = self.detector.process(frame_rgb)
        face_count = 0

        if results.detections:
            face_count = len(results.detections)
            for detection in results.detections:
                self.mp_drawing.draw_detection(frame_rgb, detection)
        
        return face_count