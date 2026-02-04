import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import argparse
import sys

class SceneDescriber:
    def __init__(self, model_path='yolov8n.pt', use_gpu=False):
        # Initialize YOLO
        print("Loading YOLO model...")
        self.yolo = YOLO(model_path)

        # Initialize EasyOCR
        print("Loading EasyOCR...")
        # gpu=False for compatibility in most environments, set to True if available
        self.reader = easyocr.Reader(['en'], gpu=use_gpu)

        # Classes that are likely to contain text
        self.text_likely_classes = [
            'cell phone', 'book', 'laptop', 'tv', 'bottle', 'cup',
            'traffic sign', 'stop sign', 'parking meter', 'bench',
            'keyboard', 'remote', 'microwave', 'oven', 'refrigerator'
        ]

        self.frame_count = 0
        self.ocr_interval = 10  # Run OCR every 10 frames
        self.last_detected_text = None

    def detect_objects(self, frame):
        """
        Detect objects in the frame using YOLO.
        """
        results = self.yolo(frame, verbose=False)
        detections = []

        # Iterate through results
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = self.yolo.names[cls]

                detections.append({
                    'label': label,
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'conf': conf
                })
        return detections

    def extract_text(self, frame, bbox, min_conf=0.5):
        """
        Extract text from a specific region (bbox) of the frame.
        """
        x1, y1, x2, y2 = bbox

        # Ensure bbox is within frame boundaries
        h, w = frame.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        if x1 >= x2 or y1 >= y2:
            return None

        roi = frame[y1:y2, x1:x2]

        # EasyOCR expects RGB or BGR? OpenCV is BGR. EasyOCR handles it but recommends RGB usually.
        # But commonly passing OpenCV image (BGR) works or converting to RGB helps accuracy.
        # Let's convert to RGB for consistency.
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

        try:
            results = self.reader.readtext(roi_rgb)
            detected_texts = []
            for (bbox, text, prob) in results:
                if prob > min_conf:
                    detected_texts.append(text)

            return " ".join(detected_texts) if detected_texts else None
        except Exception as e:
            print(f"OCR Error: {e}")
            return None

    def analyze_scene(self, detections, frame):
        """
        Analyze detections to describe the scene.
        """
        self.frame_count += 1
        should_run_ocr = (self.frame_count % self.ocr_interval == 0) or (self.frame_count == 1)

        description_parts = []

        # Sort detections by area (largest first) to find main subject
        detections.sort(key=lambda x: (x['bbox'][2]-x['bbox'][0]) * (x['bbox'][3]-x['bbox'][1]), reverse=True)

        main_subject = None
        action = None
        current_text = None

        # Find person
        person = next((d for d in detections if d['label'] == 'person'), None)

        if person:
            main_subject = "Person"
            action = "detected" # Default action

            # Check for interactions
            for d in detections:
                if d == person:
                    continue

                # Check overlap
                if self.check_overlap(person['bbox'], d['bbox']):
                    obj_label = d['label']
                    if obj_label == 'cell phone':
                        action = "holding a phone"
                    elif obj_label == 'book':
                        action = "reading a book"
                    elif obj_label == 'laptop':
                        action = "using a laptop"
                    elif obj_label == 'cup':
                        action = "drinking"
                    elif obj_label == 'bottle':
                        action = "holding a bottle"

                    # If we found an interaction object that is likely to have text, check it
                    if obj_label in self.text_likely_classes and should_run_ocr and not current_text:
                        current_text = self.extract_text(frame, d['bbox'])

        else:
            # No person, pick largest object
            if detections:
                main_subject = detections[0]['label'].capitalize()
                action = "is present"
                if detections[0]['label'] in self.text_likely_classes and should_run_ocr:
                    current_text = self.extract_text(frame, detections[0]['bbox'])
            else:
                main_subject = "Nothing"
                action = "detected"

        # Update persistent text
        if should_run_ocr:
            self.last_detected_text = current_text

        # Use cached text if valid (and if we are not in a state where everything disappeared)
        if main_subject == "Nothing":
             self.last_detected_text = None

        final_text = self.last_detected_text

        # Construct output string
        output_str = f"{main_subject} {action}"

        if final_text:
            output_str += f" (with text saying: '{final_text}')"

        return output_str

    def check_overlap(self, box1, box2):
        """
        Check if two boxes overlap significantly.
        """
        x1_a, y1_a, x2_a, y2_a = box1
        x1_b, y1_b, x2_b, y2_b = box2

        # Intersection
        xi1 = max(x1_a, x1_b)
        yi1 = max(y1_a, y1_b)
        xi2 = min(x2_a, x2_b)
        yi2 = min(y2_a, y2_b)

        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

        # Area of box2 (the object)
        box2_area = (x2_b - x1_b) * (y2_b - y1_b)

        if box2_area == 0: return False

        # If intersection covers a significant portion of the object (e.g., > 30%)
        # This implies the person and object are in the same space (holding/touching/in front)
        return (inter_area / box2_area) > 0.3

    def run(self, source="0", display=True):
        # Handle source
        if source.isdigit():
            source = int(source)

        cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            print(f"Error: Could not open video source {source}")
            return

        print("Starting video stream. Press 'q' to exit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize frame for speed if necessary, but keep reasonable resolution for OCR
            # frame = cv2.resize(frame, (640, 480))

            # Detect Objects
            detections = self.detect_objects(frame)

            # Analyze Scene
            description = self.analyze_scene(detections, frame)

            # Print output
            print(description)

            if display:
                # Visualize (Optional overlay)
                # Draw bounding boxes
                for d in detections:
                    x1, y1, x2, y2 = d['bbox']
                    label = d['label']
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Overlay text description
                cv2.putText(frame, description, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                cv2.imshow("Scene Describer", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        if display:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time Scene Description with Object Detection and OCR")
    parser.add_argument("--source", type=str, default="0", help="Video source: '0' for webcam or path to video/image file")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage for OCR")
    parser.add_argument("--no-display", action="store_true", help="Run without displaying video window")

    args = parser.parse_args()

    try:
        describer = SceneDescriber(use_gpu=not args.cpu)
        describer.run(source=args.source, display=not args.no_display)
    except KeyboardInterrupt:
        print("Interrupted by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
