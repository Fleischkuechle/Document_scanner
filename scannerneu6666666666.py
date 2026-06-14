from __future__ import annotations
import cv2
import numpy as np
from typing import Optional
from PIL import Image
import pytesseract
from flask import Flask, send_file
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
import io
import os
import threading

pytesseract.pytesseract.tesseract_cmd = os.path.join(
    os.getcwd(), "tesseract_ocr", "tesseract.exe"
)

latest_scan_path = "latest_scan.jpg"

app = Flask(__name__)


@app.route("/")
def index():
    # Browser opens the image directly
    return """
    <meta http-equiv="refresh" content="1">
    <img src="/scan.jpg" style="max-width:100%; height:auto;">
    """


@app.route("/scan.jpg")
def scan_jpg():
    if not os.path.exists(latest_scan_path):
        return "No scan yet", 404
    return send_file(latest_scan_path, mimetype="image/jpeg")


def open_camera() -> Optional[cv2.VideoCapture]:
    for backend in [0, cv2.CAP_DSHOW, cv2.CAP_VFW]:
        cap = cv2.VideoCapture(0, backend) if backend != 0 else cv2.VideoCapture(0)
        if cap.isOpened():
            return cap
    return None


def find_document_contour(image: np.ndarray) -> Optional[np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blur, 50, 150)
    edges = cv2.dilate(edges, None, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    img_area = image.shape[0] * image.shape[1]
    best = None
    best_area = 0

    for c in contours:
        area = cv2.contourArea(c)
        if area < img_area * 0.05:
            continue

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.015 * peri, True)

        if len(approx) == 4 and cv2.isContourConvex(approx):
            if area > best_area:
                best_area = area
                best = approx

    if best is not None:
        return best.reshape(4, 2).astype(np.float32)
    return None


def sort_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


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

    warped = cv2.resize(warped, (1240, 1754))
    return warped


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
    return cv2.warpAffine(
        image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


def run_scanner():
    global latest_scan_path

    cap = open_camera()
    if cap is None:
        print("No camera")
        return

    print("Press SPACE to scan, ESC to exit")
    print("Open browser: http://localhost:5000")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        contour = find_document_contour(frame)
        display = frame.copy()
        if contour is not None:
            cv2.polylines(display, [contour.astype(int)], True, (0, 255, 0), 2)

        cv2.imshow("Scanner", display)
        key = cv2.waitKey(1)

        if key == 27:
            break

        if key == 32 and contour is not None:
            warped = warp_to_a4(frame, contour)
            warped = auto_rotate(warped)

            cv2.imwrite(latest_scan_path, warped)
            print("Updated latest_scan.jpg")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    threading.Thread(
        # target=lambda: app.run(port=5000, debug=False, use_reloader=False)
        target=lambda: app.run(
            host="0.0.0.0", port=5000, debug=False, use_reloader=False
        )
    ).start()
    run_scanner()
