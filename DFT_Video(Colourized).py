import cv2
import numpy as np
import imageio.v2 as imageio
import os

# Read the Video
video_path = 'MemeFeedBot.mp4'
cap = cv2.VideoCapture(video_path)

# Create a folder to store DFT images
output_folder = 'DFT_frames'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

frame_count = 0
dft_images = []

# Loop through Frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to Grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply DFT
    dft = np.fft.fft2(gray_frame)

    # Shift the Zero-frequency Component
    dft_shifted = np.fft.fftshift(dft)

    # Compute magnitude spectrum
    magnitude_spectrum = 20 * np.log(np.abs(dft_shifted))

    # Visualize DFT Magnitude and Phase in Color
    magnitude_normalized = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX)
    magnitude_hsv = np.stack([magnitude_normalized, np.ones_like(magnitude_normalized) * 255, np.ones_like(magnitude_normalized) * 255], axis=-1)
    
    # Convert magnitude_hsv to a compatible data type before converting to RGB
    magnitude_hsv_uint8 = (magnitude_hsv * 255).astype(np.uint8)
    magnitude_rgb = cv2.cvtColor(magnitude_hsv_uint8, cv2.COLOR_HSV2RGB)

    # Save DFT Images
    dft_image_path = os.path.join(output_folder, f'dft_frame_{frame_count}.jpg')
    cv2.imwrite(dft_image_path, magnitude_rgb)

    # Append DFT image path to list for GIF creation
    dft_images.append(dft_image_path)

    frame_count += 1

cap.release()

# Compile DFT Images into GIF
gif_path = 'dft_animation_color.gif'
with imageio.get_writer(gif_path, mode='I') as writer:
    for dft_image_path in dft_images:
        image = imageio.imread(dft_image_path)
        writer.append_data(image)

# Clean up: Remove DFT images after creating GIF
for dft_image_path in dft_images:
    os.remove(dft_image_path)

print(f'GIF saved at: {gif_path}')
