import cv2
import numpy as np
import imageio
import os

# 1. Read the Video
video_path = 'upscaled.mp4'
cap = cv2.VideoCapture(video_path)

# Create a folder to store DFT images
output_folder = 'DFT_frames'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

frame_count = 0
dft_images = []

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

    # 7. Save DFT Images
    magnitude_spectrum = 20 * np.log(np.abs(dft_shifted))
    dft_image_path = os.path.join(output_folder, f'dft_frame_{frame_count}.jpg')
    cv2.imwrite(dft_image_path, magnitude_spectrum)

    # Append DFT image path to list for GIF creation
    dft_images.append(dft_image_path)

    frame_count += 1

cap.release()

# 8. Compile DFT Images into GIF
gif_path = 'dft_animation.gif'
with imageio.get_writer(gif_path, mode='I') as writer:
    for dft_image_path in dft_images:
        image = imageio.imread(dft_image_path)
        writer.append_data(image)

# Clean up: Remove DFT images after creating GIF
for dft_image_path in dft_images:
    os.remove(dft_image_path)

print(f'GIF saved at: {gif_path}')