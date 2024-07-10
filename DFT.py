import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Read the Image
image = cv2.imread('F6E67DdWMAAy5Tz.jpeg', cv2.IMREAD_GRAYSCALE)

# 2. Convert to Grayscale
# If the image is already grayscale, skip this step

# 3. DFT
dft = np.fft.fft2(image)

# 4. Shift the Zero-frequency Component
dft_shifted = np.fft.fftshift(dft)

# 5. Visualize the Spectrum
magnitude_spectrum = 20 * np.log(np.abs(dft_shifted))

plt.figure(figsize=(8, 8))
plt.subplot(121), plt.imshow(image, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()
