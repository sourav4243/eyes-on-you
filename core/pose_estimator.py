import mediapipe as mp

class PoseEstimator:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def process(self, frame_rgb, face_count):
        """Returns True if the user is looking away, False otherwise."""
        if face_count !=1:
            return False

        results = self.mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            nose_x = landmarks[0].x         # nose
            left_x = landmarks[234].x       # left cheek    
            right_x = landmarks[454].x      # right cheek

            dist_left = abs(nose_x - left_x)
            dist_right = abs(nose_x - right_x)

            if(dist_left + dist_right) > 0:
                ratio = dist_left / (dist_left + dist_right)
                # extreme ratio means head is turned
                if ratio < 0.25 or ratio > 0.75:
                    return True
                
        return False