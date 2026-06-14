import cv2
import numpy as np
import os
from imutils.perspective import four_point_transform
import pytesseract

# --- Setup Output Directory ---
if not os.path.exists("output"):
    os.makedirs("output")

# --- Configuration ---
pytesseract.pytesseract.tesseract_cmd = os.path.join(
    os.getcwd(), "tesseract_ocr", "tesseract.exe"
)


# --- Camera Selection ---
def select_camera():
    available_indices = []
    for i in range(5):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            available_indices.append(i)
            cap.release()
    return available_indices[0] if available_indices else None


cam_index = select_camera()
if cam_index is None:
    exit()

cap = cv2.VideoCapture(cam_index + cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Window Setup
for name in ["Input", "Warped", "Processed"]:
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, 640, 480)

document_contour = np.array([[0, 0], [1920, 0], [1920, 1080], [0, 1080]])


def force_a4_format(image):
    # A4 standard: 2480x3508 pixels
    return cv2.resize(image, (2480, 3508), interpolation=cv2.INTER_CUBIC)


def enhance_lighting(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)


def image_processing(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Median Blur + Adaptive Thresholding for clean text
    denoised = cv2.medianBlur(gray, 3)
    return cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 8
    )


def scan_detection(image):
    global document_contour
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, threshold = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.015 * peri, True)
            if len(approx) == 4:
                document_contour = approx
                break


def center_text(image, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, 2, 5)[0]
    cv2.putText(
        image,
        text,
        ((image.shape[1] - text_size[0]) // 2, (image.shape[0] + text_size[1]) // 2),
        font,
        2,
        (255, 0, 255),
        5,
        cv2.LINE_AA,
    )


# --- Main Loop ---
count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.rotate(frame, cv2.ROTATE_180)
    frame_copy = frame.copy()
    scan_detection(frame_copy)
    cv2.drawContours(frame_copy, [document_contour], -1, (0, 255, 0), 3)

    # Transform and Format
    warped = four_point_transform(frame_copy, document_contour.reshape(4, 2))
    warped = force_a4_format(warped)
    warped_enhanced = enhance_lighting(warped)
    processed = image_processing(warped_enhanced)

    cv2.imshow("Input", frame_copy)
    cv2.imshow("Warped", cv2.resize(warped_enhanced, (400, 565)))  # Scaled for window
    cv2.imshow("Processed", cv2.resize(processed, (400, 565)))

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key == ord("s"):
        cv2.imwrite(f"output/scanned_{count}.jpg", warped_enhanced)
        center_text(frame_copy, "Scan Saved")
        cv2.imshow("Input", frame_copy)
        cv2.waitKey(500)
        count += 1
    elif key == ord("o") and count > 0:
        ocr_text = pytesseract.image_to_string(warped_enhanced)
        with open(f"output/recognized_{count - 1}.txt", "w", encoding="utf-8") as f:
            f.write(ocr_text)
        center_text(frame_copy, "Text Saved")
        cv2.waitKey(500)

cap.release()
cv2.destroyAllWindows()
