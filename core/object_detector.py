from ultralytics import YOLO
import cv2

class ObjectDetector:
    def __init__(self):
        # download tiny CPU-friendly YOLOv8 model 
        self.model = YOLO('yolov8n.pt')
        self.frame_skip = 5
        self.frame_count = 0
        self.is_holding_object = False

        # memory: store last known bounding boxes
        self.last_boxes = []

    def process(self, frame_rgb):
        self.frame_count += 1

        # run yolo on evey 5th frame to reduce CPU load
        if self.frame_count % self.frame_skip == 0:
            self.is_holding_object = False
            self.last_boxes = []

            results = self.model(frame_rgb, verbose=False)

            for r in results:
                for box in r.boxes:
                    class_id = int(box.cls[0])
                    # COCO dataset classes: 67 = cell phone, 73 = book
                    if class_id ==67 or class_id == 73:
                        self.is_holding_object = True

                        # save coordinated into memory list
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        self.last_boxes.append((x1, y1, x2, y2))
            
        # drawing boxes from memory, even on skipped frames:
        for (x1, y1, x2, y2) in self.last_boxes:
            cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (255, 0, 0), 3)
            cv2.putText(frame_rgb, "UNAUTHORIZED OBJECT", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        return self.is_holding_object