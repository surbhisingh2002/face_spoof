import face_recognition
import cv2
import numpy as np

def match_faces(image_path_1, image_path_2):

    # ── Load both images ──
    img1 = face_recognition.load_image_file(image_path_1)
    img2 = face_recognition.load_image_file(image_path_2)

    # ── Get face encodings ──
    encodings1 = face_recognition.face_encodings(img1)
    encodings2 = face_recognition.face_encodings(img2)

    # ── Check if face found in both ──
    if len(encodings1) == 0:
        print(" No face found in image 1")
        return

    if len(encodings2) == 0:
        print(" No face found in image 2")
        return

    enc1 = encodings1[0]
    enc2 = encodings2[0]

    # ── Calculate distance (lower = more similar) ──
    distance = face_recognition.face_distance([enc1], enc2)[0]

    # ── Convert distance to percentage ──
    match_percent = (1 - distance) * 100

    # ── Result ──
    print("\n======================")
    print(f"Match Percentage : {match_percent:.2f}%")
    print(f"Distance         : {distance:.4f}")

    if match_percent >= 50:
        print("Result           : SAME PERSON")
    elif 30 <= match_percent < 50:
        print("Result           : PERSON IN REVIEW")
    else:
        print("Result           : DIFFERENT PERSON")
    print("======================\n")

    return match_percent


# ── Change these to your image paths ──
image1 = "images/wmn1.PNG"   
image2 = "images/wmnn2.PNG"   

match_faces(image1, image2)
