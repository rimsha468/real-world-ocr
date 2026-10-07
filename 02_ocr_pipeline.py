import os

import cv2
import pytesseract


# Find Tesseract
tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

if os.path.exists(tesseract_path):
    pytesseract.pytesseract.tesseract_cmd = tesseract_path


base_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(base_dir, "images")
output_dir = os.path.join(base_dir, "outputs")

image_names = ["image1.png", "image2.png", "image3.png"]


def preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, otsu = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    return {
        "Grayscale": gray,
        "Otsu Threshold": otsu,
        "Adaptive Threshold": adaptive
    }


def get_confidence(image):
    data = pytesseract.image_to_data(
        image,
        output_type=pytesseract.Output.DICT
    )

    scores = []

    for value in data["conf"]:
        value = float(value)

        if value >= 0:
            scores.append(value)

    if not scores:
        return 0

    return sum(scores) / len(scores)


def clean_text(text):
    lines = []

    for line in text.splitlines():
        line = line.strip()

        if line:
            lines.append(line)

    return "\n".join(lines)


def run_ocr(image):
    results = {}

    processed_images = preprocess(image)

    for method, processed_image in processed_images.items():
        confidence = get_confidence(processed_image)
        text = pytesseract.image_to_string(processed_image)
        text = clean_text(text)

        results[method] = {
            "confidence": confidence,
            "text": text
        }

    best_method = max(
        results,
        key=lambda method: results[method]["confidence"]
    )

    return best_method, results


def save_result(image_name, method, confidence, text):
    output_name = os.path.splitext(image_name)[0] + ".txt"
    output_path = os.path.join(output_dir, output_name)

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(f"Image: {image_name}\n")
        file.write(f"Best preprocessing: {method}\n")
        file.write(f"Confidence: {confidence:.2f}%\n\n")
        file.write(text)

    return output_path


def main():
    os.makedirs(output_dir, exist_ok=True)

    for image_name in image_names:
        image_path = os.path.join(images_dir, image_name)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Could not read {image_name}")
            continue

        best_method, results = run_ocr(image)

        best_confidence = results[best_method]["confidence"]
        best_text = results[best_method]["text"]

        output_path = save_result(
            image_name,
            best_method,
            best_confidence,
            best_text
        )

        print("=" * 60)
        print(f"IMAGE: {image_name}")
        print("=" * 60)

        for method, result in results.items():
            print(
                f"{method}: "
                f"{result['confidence']:.2f}%"
            )

        print(
            f"\nSelected method: {best_method} "
            f"({best_confidence:.2f}%)"
        )

        print(f"Saved to: {output_path}")

        print("\n--- OCR TEXT ---")
        print(best_text)
        print()


if __name__ == "__main__":
    main()
