from __future__ import annotations
import cv2
import numpy as np
from typing import Optional
from PIL import Image
import pytesseract
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
import io
import os

# --- Configuration ---
pytesseract.pytesseract.tesseract_cmd = os.path.join(
    os.getcwd(), "tesseract_ocr", "tesseract.exe"
)


# ---------------------------------------------------------
# Kamera robust öffnen (MSMF → DirectShow → VFW)
# ---------------------------------------------------------
def open_camera() -> Optional[cv2.VideoCapture]:
    for backend in [0, cv2.CAP_DSHOW, cv2.CAP_VFW]:
        cap = cv2.VideoCapture(0, backend) if backend != 0 else cv2.VideoCapture(0)
        if cap.isOpened():
            return cap
    return None


# ---------------------------------------------------------
# 1. Balanced document contour detection
# ---------------------------------------------------------
def find_document_contour(image: np.ndarray) -> Optional[np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    edges = cv2.Canny(blur, 50, 150)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    img_area = image.shape[0] * image.shape[1]
    best_contour = None
    best_area = 0

    for c in contours:
        area = cv2.contourArea(c)
        if area < img_area * 0.05:
            continue

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.015 * peri, True)

        if len(approx) != 4:
            continue

        if not cv2.isContourConvex(approx):
            continue

        pts = approx.reshape(4, 2)
        w = np.linalg.norm(pts[0] - pts[1])
        h = np.linalg.norm(pts[1] - pts[2])
        if min(w, h) == 0:
            continue

        ratio = max(w, h) / min(w, h)

        if 1.0 < ratio < 2.0:
            if area > best_area:
                best_area = area
                best_contour = approx

    if best_contour is not None:
        return best_contour.reshape(4, 2).astype(np.float32)

    return None


# ---------------------------------------------------------
# 2. Sort points (top-left, top-right, bottom-right, bottom-left)
# ---------------------------------------------------------
def sort_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left

    return rect


# ---------------------------------------------------------
# 3. Warp perspective to document, then scale to A4 (150 DPI)
# ---------------------------------------------------------
def warp_to_a4(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = sort_points(pts)

    (tl, tr, br, bl) = pts

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    dst = np.array(
        [[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]],
        dtype="float32",
    )

    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # Scale to A4 ratio (150 DPI)
    a4_w, a4_h = 1240, 1754
    warped = cv2.resize(warped, (a4_w, a4_h))

    return warped


# ---------------------------------------------------------
# 4. Auto-rotate using OpenCV (no OSD)
# ---------------------------------------------------------
def auto_rotate(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)

    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    coords = np.column_stack(np.where(thresh > 0))
    if coords.size == 0:
        return image

    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(
        image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )

    return rotated


# ---------------------------------------------------------
# 5. Save PDF with OCR layer
# ---------------------------------------------------------
def save_pdf_with_ocr(image: np.ndarray, filename: str) -> None:
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    text = pytesseract.image_to_string(pil_img)

    packet = io.BytesIO()
    c = canvas.Canvas(packet, pagesize=A4)

    img_bytes = io.BytesIO()
    pil_img.save(img_bytes, format="PNG")
    img_bytes.seek(0)

    img_reader = ImageReader(img_bytes)
    c.drawImage(img_reader, 0, 0, width=A4[0], height=A4[1])

    c.setFillColorRGB(1, 1, 1, alpha=0)
    c.setFont("Helvetica", 8)

    y = 820
    for line in text.split("\n"):
        if not line.strip():
            continue
        c.drawString(20, y, line)
        y -= 10

    c.save()

    with open(filename, "wb") as f:
        f.write(packet.getvalue())


# ---------------------------------------------------------
# 6. Main scanning loop
# ---------------------------------------------------------
def run_scanner() -> None:
    cap = open_camera()
    if cap is None:
        print("❌ Keine Kamera verfügbar oder Zugriff verweigert.")
        return

    print("Press SPACE to scan, ESC to exit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Kamera liefert keine Frames.")
            break

        display = frame.copy()
        contour = find_document_contour(frame)

        if contour is not None:
            cv2.polylines(display, [contour.astype(int)], True, (0, 255, 0), 2)

        cv2.imshow("Document Scanner", display)
        key = cv2.waitKey(1)

        if key == 27:
            break

        if key == 32:
            if contour is None:
                print("⚠️ Kein Dokument erkannt.")
                continue

            warped = warp_to_a4(frame, contour)
            warped = auto_rotate(warped)

            preview = cv2.resize(warped, None, fx=0.3, fy=0.3)

            cv2.imshow("A4 Preview", preview)

            save_pdf_with_ocr(warped, "scan.pdf")
            print("✅ Gespeichert als scan.pdf")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_scanner()
