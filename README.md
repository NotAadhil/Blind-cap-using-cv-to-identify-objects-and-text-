

Project:  AI Smart Cap
Project Overview: This is an assistive technology prototype designed for visually impaired individuals. It uses a camera mounted on a cap to provide real-time auditory feedback about the user's surroundings. The system is built using Python and leverages deep learning for object detection and character recognition.

Key Features:

Object Detection: Uses YOLOv8 to identify people, chairs, bottles, and other obstacles.

Safety Alerts: Automatically warns the user if an object is too close (distance estimation based on screen area).

Smart OCR: Dual-pass text reading system using EasyOCR with image upscaling and adaptive thresholding.

Center Focus: Only announces objects in the center 60 percent of the frame to reduce noise.

Audio Queue: Multi-threaded speech engine to prevent lag and overlapping audio.

Hardware Used:

Laptop (Intel i5-1335U)

USB Webcam mounted on a cap

Standard Headphones

Software and Libraries:

Python 3.10+

OpenCV (Image processing)

Ultralytics YOLOv8 (Vision AI)

EasyOCR (Text recognition)

pyttsx3 (Text-to-speech)

How to Run:

Install requirements: pip install -r requirements.txt

Connect webcam.

Run: python main.py

Use Spacebar to trigger text reading.

Press q while focusing on the video window to exit.

Exhibition Goals: This project demonstrates the potential of computer vision in accessibility. Future versions will be ported to a Raspberry Pi for a fully wireless experience.
