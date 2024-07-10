import cv2
import numpy as np
import os

# 1. Read the Video
video_path = 'MemeFeedBot.mp4'
cap = cv2.VideoCapture(video_path)

# Define the codec and create VideoWriter object
output_video_path = 'dft_video_1.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size, False)  # Set isColor flag to False for grayscale

# 3. Loop through Frames 
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 4. Convert to Grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 5. Apply DFT
    dft = np.fft.fft2(gray_frame)

    # 6. Shift the Zero-frequency Component
    dft_shifted = np.fft.fftshift(dft)

    # 7. Compute Magnitude Spectrum
    magnitude_spectrum = 20 * np.log(np.abs(dft_shifted))

    # 8. Write Frame to Video
    out.write(np.uint8(magnitude_spectrum))

cap.release()
out.release()

print(f'Video saved at: {output_video_path}')
