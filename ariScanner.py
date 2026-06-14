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

    # Strong edges but suppress small details
    edges = cv2.Canny(blur, 50, 150)

    # Connect broken edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    img_area = image.shape[0] * image.shape[1]
    best_contour = None
    best_area = 0

    for c in contours:
        area = cv2.contourArea(c)
        if area < img_area * 0.05:  # only 5% minimum area
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

        if 1.0 < ratio < 2.0:  # tolerant A4 ratio
            if area > best_area:
                best_area = area
                best_contour = approx

    if best_contour is not None:
        return best_contour.reshape(4, 2).astype(np.float32)

    return None


# ---------------------------------------------------------
# 2. Warp perspective to A4 (150 DPI)
# ---------------------------------------------------------
def warp_to_a4(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    width, height = 1240, 1754  # A4 @ 150 DPI

    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )

    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(image, M, (width, height))

    return warped


# ---------------------------------------------------------
# 3. Save PDF with OCR layer
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
# 4. Main scanning loop
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

        if key == 27:  # ESC
            break

        if key == 32:  # SPACE
            if contour is None:
                print("⚠️ Kein Dokument erkannt.")
                continue

            warped = warp_to_a4(frame, contour)

            # Live preview of the scanned A4
            cv2.imshow("A4 Preview", warped)

            save_pdf_with_ocr(warped, "scan.pdf")
            print("✅ Gespeichert als scan.pdf")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_scanner()
