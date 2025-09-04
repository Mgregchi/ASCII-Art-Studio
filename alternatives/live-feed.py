import cv2
from PIL import Image
import numpy as np

# Load video
video = cv2.VideoCapture("input.mp4")
ascii_chars = list("@%#*+=-:. ")

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break
    
    # Convert to grayscale and resize
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (80, 40))
    
    # Map pixels to ASCII
    ascii_frame = ""
    for row in resized:
        for pixel in row:
            ascii_frame += ascii_chars[pixel // 32]  # Map 0-255 to 0-7
        ascii_frame += "\n"
    
    # Display in terminal (or save for rendering)
    print("\033[H\033[J")  # Clear terminal
    print(ascii_frame)

video.release()