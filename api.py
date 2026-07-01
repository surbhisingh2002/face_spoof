from fastapi import FastAPI, UploadFile, File
import face_recognition
import numpy as np
import shutil
import os
from fastapi import FastAPI

app = FastAPI()
def match_faces(image_path_1, image_path_2):

    img1 = face_recognition.load_image_file(image_path_1)
    img2 = face_recognition.load_image_file(image_path_2)

    enc1 = face_recognition.face_encodings(img1)
    enc2 = face_recognition.face_encodings(img2)

    if len(enc1) == 0 or len(enc2) == 0:
        return {"error": "Face not found in one or both images"}

    enc1 = enc1[0]
    enc2 = enc2[0]

    distance = face_recognition.face_distance([enc1], enc2)[0]
    match_percent = (1 - distance) * 100

    result = (
        "SAME PERSON" if match_percent >= 50 else
        "PERSON IN REVIEW" if match_percent >= 30 else
        "DIFFERENT PERSON"
    )

    return {
        "match_percentage": round(match_percent, 2),
        "distance": float(distance),
        "result": result
    }


@app.post("/compare-faces/")
async def compare_faces(image1: UploadFile = File(...),
                         image2: UploadFile = File(...)):

    path1 = f"temp_{image1.filename}"
    path2 = f"temp_{image2.filename}"

    with open(path1, "wb") as f:
        shutil.copyfileobj(image1.file, f)

    with open(path2, "wb") as f:
        shutil.copyfileobj(image2.file, f)

    result = match_faces(path1, path2)

    os.remove(path1)
    os.remove(path2)

    return result