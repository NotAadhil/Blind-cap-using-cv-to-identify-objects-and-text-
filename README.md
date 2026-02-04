# Real-time Scene Description with Object Detection and OCR

This application describes the scene captured by the webcam in real-time, combining Object Detection (YOLOv8) and Optical Character Recognition (OCR) (EasyOCR).

It outputs descriptions like: `Person holding a phone (with text saying: 'Hello World')`

## Requirements

- Python 3.8+
- Webcam

## Installation

1.  Clone the repository (or download the files).
2.  Install the required dependencies:

    ```bash
    pip install opencv-python ultralytics easyocr
    ```

    Note: `easyocr` and `ultralytics` will install `torch` (PyTorch). If you have a GPU, ensure you install the appropriate CUDA-enabled PyTorch version for better performance.

## Usage

Run the application:

```bash
python scene_describer.py
```

### Options

- `--source`: Video source. Default is `0` (webcam). You can provide a path to a video file or image.
  ```bash
  python scene_describer.py --source video.mp4
  ```
- `--cpu`: Force CPU usage for OCR (useful if you don't have a CUDA GPU or want to save VRAM).
  ```bash
  python scene_describer.py --cpu
  ```
- `--no-display`: Run without the video window (output only to console).
  ```bash
  python scene_describer.py --no-display
  ```

## How it works

1.  **Object Detection**: Uses YOLOv8 (Nano model) to detect objects (Person, Cell phone, Book, etc.).
2.  **Scene Logic**: Checks for interactions (e.g., overlapping bounding boxes) to infer actions (e.g., "Person holding a phone").
3.  **OCR**: If an object that typically contains text is detected (and interacts with the person or is the main subject), EasyOCR extracts text from that region.
4.  **Output**: Formats the detections and text into a descriptive string.

## Customization

You can modify the `SceneDescriber` class in `scene_describer.py` to add more interaction rules or change the confidence thresholds.
