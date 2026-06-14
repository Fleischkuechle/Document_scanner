from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from imutils.perspective import four_point_transform
import pytesseract
import io

app = FastAPI()

# CORS erlauben, damit der Browser auf den Server zugreifen darf
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
import pytesseract
import os

# Setze den Pfad dynamisch auf den Ordner tesseract_ocr im Projektverzeichnis
pytesseract.pytesseract.tesseract_cmd = os.path.join(
    os.getcwd(), "tesseract_ocr", "tesseract.exe"
)


def scan_detection(image):
    # Einfache adaptive Schwellenwert-Logik zur Konturerkennung
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, threshold = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4:
            return approx  # Gefundenes Dokument
    return None


@app.post("/process-scan")
async def process_scan(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 1. Dokument finden & entzerren
    contour = scan_detection(image)
    if contour is not None:
        warped = four_point_transform(image, contour.reshape(4, 2))
    else:
        warped = image  # Fallback, falls nichts gefunden

    # 2. OCR
    ocr_text = pytesseract.image_to_string(warped, lang="deu")

    # 3. Rückgabe
    return {"text": ocr_text}


from fastapi.responses import FileResponse


# Diese Route sorgt dafür, dass beim Aufruf von http://127.0.0.1:8000/
# deine index.html geladen wird
@app.get("/")
async def get_index():
    return FileResponse("index.html")
