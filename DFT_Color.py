import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the colored image
image = cv2.imread('F6E67DdWMAAy5Tz.jpeg')

# Split the image into color channels
b, g, r = cv2.split(image)

# Apply DFT to each channel
dft_b = cv2.dft(np.float32(b), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_g = cv2.dft(np.float32(g), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_r = cv2.dft(np.float32(r), flags=cv2.DFT_COMPLEX_OUTPUT)

# Compute magnitude spectrum for visualization
magnitude_spectrum_b = 20 * np.log(cv2.magnitude(dft_b[:, :, 0], dft_b[:, :, 1]))
magnitude_spectrum_g = 20 * np.log(cv2.magnitude(dft_g[:, :, 0], dft_g[:, :, 1]))
magnitude_spectrum_r = 20 * np.log(cv2.magnitude(dft_r[:, :, 0], dft_r[:, :, 1]))

# Visualize the magnitude spectrum of each channel
plt.figure(figsize=(10, 10))
plt.subplot(131), plt.imshow(magnitude_spectrum_b, cmap='gray')
plt.title('Blue Channel DFT'), plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(magnitude_spectrum_g, cmap='gray')
plt.title('Green Channel DFT'), plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(magnitude_spectrum_r, cmap='gray')
plt.title('Red Channel DFT'), plt.xticks([]), plt.yticks([])
plt.show()
