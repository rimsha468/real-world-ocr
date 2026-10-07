import os

import cv2
import pytesseract
import matplotlib.pyplot as plt


# Find Tesseract
tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

if os.path.exists(tesseract_path):
    pytesseract.pytesseract.tesseract_cmd = tesseract_path


# Load image
image = cv2.imread("images/image1.png")

if image is None:
    print("Could not load the image.")
    exit()


# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# Blur the image before thresholding
blur = cv2.GaussianBlur(gray, (5, 5), 0)


# Apply Otsu thresholding
threshold = cv2.threshold(
    blur,
    0,
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)[1]


# Resize the grayscale image
resized = cv2.resize(
    gray,
    None,
    fx=2,
    fy=2,
    interpolation=cv2.INTER_CUBIC
)


# Run OCR on each version
text_gray = pytesseract.image_to_string(gray)
text_threshold = pytesseract.image_to_string(threshold)
text_resized = pytesseract.image_to_string(resized)


# Show OCR results
print("========== GRAYSCALE ==========")
print(text_gray)

print("\n========== BLUR + THRESHOLD ==========")
print(text_threshold)

print("\n========== RESIZED ==========")
print(text_resized)


# Display the different image versions
plt.figure(figsize=(16, 5))

plt.subplot(1, 4, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original")
plt.axis("off")

plt.subplot(1, 4, 2)
plt.imshow(gray, cmap="gray")
plt.title("Grayscale")
plt.axis("off")

plt.subplot(1, 4, 3)
plt.imshow(threshold, cmap="gray")
plt.title("Blur + Threshold")
plt.axis("off")

plt.subplot(1, 4, 4)
plt.imshow(resized, cmap="gray")
plt.title("Resized")
plt.axis("off")

plt.tight_layout()
plt.show()
