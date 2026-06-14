from __future__ import annotations
import cv2
import numpy as np
from typing import Optional, Tuple
from PIL import Image
import pytesseract
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
import io


# ---------------------------------------------------------
# 1. Find document contour
# ---------------------------------------------------------
def find_document_contour(image: np.ndarray) -> Optional[np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 75, 200)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)

    return None


# ---------------------------------------------------------
# 2. Warp perspective to A4
# ---------------------------------------------------------
def warp_to_a4(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    width, height = 2480, 3508  # A4 @ 300 DPI

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

    # OCR
    text = pytesseract.image_to_string(pil_img)

    # PDF buffer
    packet = io.BytesIO()
    c = canvas.Canvas(packet, pagesize=A4)

    # Insert scanned image
    img_bytes = io.BytesIO()
    pil_img.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    c.drawImage(img_bytes, 0, 0, width=A4[0], height=A4[1])

    # Invisible OCR text layer
    c.setFillColorRGB(1, 1, 1, alpha=0)
    c.setFont("Helvetica", 8)

    y = 820
    for line in text.split("\n"):
        c.drawString(20, y, line)
        y -= 10

    c.save()

    with open(filename, "wb") as f:
        f.write(packet.getvalue())


# ---------------------------------------------------------
# 4. Main scanning loop
# ---------------------------------------------------------
def run_scanner() -> None:
    cap = cv2.VideoCapture(1)

    print("Press SPACE to scan, ESC to exit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera error")
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
                print("No document detected")
                continue

            warped = warp_to_a4(frame, contour)
            save_pdf_with_ocr(warped, "scan.pdf")
            print("Saved scan.pdf")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_scanner()
