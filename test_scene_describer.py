import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import cv2
from scene_describer import SceneDescriber

class TestSceneDescriber(unittest.TestCase):

    @patch('scene_describer.YOLO')
    @patch('scene_describer.easyocr.Reader')
    def setUp(self, mock_reader, mock_yolo):
        self.describer = SceneDescriber()
        # Mock detections
        self.describer.yolo = MagicMock()
        self.describer.reader = MagicMock()

    def test_person_holding_book_with_text(self):
        # Mock detections: Person and Book overlapping
        # Frame size 100x100
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        detections = [
            {'label': 'person', 'bbox': [10, 10, 90, 90], 'conf': 0.9},
            {'label': 'book', 'bbox': [40, 40, 60, 60], 'conf': 0.8}
        ]

        # Mock OCR result for the book area
        # readtext returns list of (bbox, text, prob)
        self.describer.reader.readtext.return_value = [([], "Python Guide", 0.9)]

        # We need to mock extract_text because it calls reader.readtext with cropped image
        # Actually, let's just let it call the mock reader.

        description = self.describer.analyze_scene(detections, frame)

        expected = "Person reading a book (with text saying: 'Python Guide')"
        self.assertEqual(description, expected)

    def test_person_holding_phone_no_text(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        detections = [
            {'label': 'person', 'bbox': [10, 10, 90, 90], 'conf': 0.9},
            {'label': 'cell phone', 'bbox': [40, 40, 60, 60], 'conf': 0.8}
        ]

        self.describer.reader.readtext.return_value = [] # No text found

        description = self.describer.analyze_scene(detections, frame)

        expected = "Person holding a phone"
        self.assertEqual(description, expected)

    def test_no_person_just_object(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        detections = [
            {'label': 'laptop', 'bbox': [20, 20, 80, 80], 'conf': 0.8}
        ]

        self.describer.reader.readtext.return_value = [([], "Dell", 0.9)]

        description = self.describer.analyze_scene(detections, frame)

        expected = "Laptop is present (with text saying: 'Dell')"
        self.assertEqual(description, expected)

    def test_nothing_detected(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        detections = []

        description = self.describer.analyze_scene(detections, frame)

        expected = "Nothing detected"
        self.assertEqual(description, expected)

if __name__ == '__main__':
    unittest.main()
