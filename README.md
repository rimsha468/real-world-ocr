# Real-World OCR Pipeline

A simple computer vision project for extracting text from real-world images using **Tesseract OCR** and comparing different image preprocessing techniques.

## Overview

OCR performance can vary depending on the quality and characteristics of an image. In this project, I tested three preprocessing approaches before applying OCR:

* Grayscale conversion
* Otsu thresholding
* Adaptive thresholding

The pipeline processes multiple images, calculates Tesseract's average OCR confidence for each preprocessing method, selects the method with the highest confidence, and saves the resulting text.

## Project Structure

```text
01-real-world-ocr/
│
├── 01_ocr_fundamentals.py
├── 02_ocr_pipeline.py
│
├── images/
│   ├── image1.png
│   ├── image2.png
│   └── image3.png
│
├── outputs/
│   ├── image1.txt
│   ├── image2.txt
│   └── image3.txt
│
└── README.md
```

## Technologies

* Python
* OpenCV
* Tesseract OCR
* Pytesseract
* NumPy
* Matplotlib

## How It Works

The pipeline follows these steps:

1. Load an input image.
2. Convert the image to grayscale.
3. Create an Otsu-thresholded version.
4. Create an adaptive-thresholded version.
5. Run OCR on each version.
6. Calculate the average OCR confidence.
7. Compare the preprocessing methods.
8. Select the method with the highest confidence.
9. Save the selected OCR text to the `outputs` folder.

## Experiment Results

Three different types of images were tested: a product label, a receipt, and a structured document.

| Image               | Grayscale | Otsu Threshold | Adaptive Threshold | Selected  |
| ------------------- | --------: | -------------: | -----------------: | --------- |
| Product label       |    81.12% |         76.66% |             69.74% | Grayscale |
| Receipt             |    90.56% |         89.88% |             87.55% | Grayscale |
| Structured document |    93.76% |         93.09% |             92.73% | Grayscale |

For these three images, grayscale produced the highest average OCR confidence in every case.

## Observation

The experiment showed that preprocessing does not affect every image in the same way.

For example, thresholding correctly recognized **"Tomato Basil Soup"** in the product-label image, while the grayscale version produced **"Tomato Rasil Soup"**. However, grayscale still had a higher overall confidence score.

This highlights an important limitation of using OCR confidence as the only evaluation measure:

> A higher confidence score does not necessarily mean that every recognized word is correct.

Therefore, preprocessing methods should be evaluated using both confidence scores and actual OCR output.

## Running the Project

Install the required Python packages:

```bash
pip install opencv-python numpy matplotlib pillow pytesseract
```

Tesseract OCR must also be installed separately on the system.

After installation, update the Tesseract path in `02_ocr_pipeline.py` if necessary.

Run:

```bash
python 02_ocr_pipeline.py
```

The extracted text will be saved automatically in the `outputs` folder.

## What I Learned

* How OCR systems process image input.
* How grayscale conversion affects OCR.
* The difference between global and adaptive thresholding.
* How to use Tesseract's OCR confidence information.
* How preprocessing can affect recognition differently across image types.
* Why confidence scores should not be treated as a direct measurement of actual OCR accuracy.

## Future Improvements

Possible extensions include:

* Image resizing and denoising.
* Automatic document orientation detection.
* Better handling of tables and structured documents.
* Word-level confidence analysis.
* Evaluation against manually verified ground-truth text.
* Testing additional OCR engines.
