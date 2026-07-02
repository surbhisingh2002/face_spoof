from fastapi import FastAPI, UploadFile, File
import numpy as np
import face_recognition
import uvicorn

app = FastAPI()


def read_image(file):
    image = face_recognition.load_image_file(file)
    return image


def match_faces(img1, img2):

    enc1 = face_recognition.face_encodings(img1)
    enc2 = face_recognition.face_encodings(img2)

    if len(enc1) == 0:
        return {"error": "No face found in image 1"}

    if len(enc2) == 0:
        return {"error": "No face found in image 2"}

    enc1 = enc1[0]
    enc2 = enc2[0]

    distance = face_recognition.face_distance([enc1], enc2)[0]
    match_percent = (1 - distance) * 100

    result = "SAME PERSON" if match_percent >= 50 else "DIFFERENT PERSON"

    return {
        "match_percent": round(match_percent, 2),
        "result": result
    }


@app.post("/match")
async def match(file1: UploadFile = File(...), file2: UploadFile = File(...)):

    img1 = read_image(file1.file)
    img2 = read_image(file2.file)

    return match_faces(img1, img2)


@app.get("/health")
def health_check():
    return {"status": "healthy"}
