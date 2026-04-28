# import cv2
# import numpy as np
# import mediapipe as mp

# # ---------------- INIT ---------------- #
# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

# def detect_face_landmarks(image):
#     rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     results = face_mesh.process(rgb)
#     return results.multi_face_landmarks

# def fake_or_real(image_path):
#     image = cv2.imread(image_path)
#     h, w = image.shape[:2]

#     faces = detect_face_landmarks(image)

#     if not faces:
#         return "NO FACE DETECTED "

#     lm = faces[0].landmark

#     # ---------------- SIMPLE LIVENESS HEURISTICS ---------------- #

#     # Eye openness check
#     left_eye_top = lm[159].y
#     left_eye_bottom = lm[145].y

#     right_eye_top = lm[386].y
#     right_eye_bottom = lm[374].y

#     left_eye_open = abs(left_eye_top - left_eye_bottom)
#     right_eye_open = abs(right_eye_top - right_eye_bottom)

#     avg_eye_open = (left_eye_open + right_eye_open) / 2

#     # Nose stability check (flat image = too stable)
#     nose = lm[1]
#     nose_variation = abs(nose.z)

#     # ---------------- DECISION RULE ---------------- #

#     if avg_eye_open < 0.005:
#         return "FAKE (No natural eye structure / photo likely) "

#     if nose_variation < 0.01:
#         return "FAKE (Flat depth detected - possible photo/screen) "

#     return "REAL (Likely live face) "

# # ---------------- RUN ---------------- #
# image_path = "images/1000063515.jpg"   # change your image here
# result = fake_or_real(image_path)


# print("\nRESULT:", result)

import cv2
import numpy as np
import mediapipe as mp

# ---------------- INIT ---------------- #
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

def detect_face_landmarks(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    return results.multi_face_landmarks

def get_liveness_percentage(image_path):

    # Load image
    image = cv2.imread(image_path)

    # Safety check
    if image is None:
        return "ERROR: Image not found "

    h, w = image.shape[:2]

    # Detect face
    faces = detect_face_landmarks(image)
    if not faces:
        return "NO FACE DETECTED "

    lm = faces[0].landmark
    print(lm)

    # ---------------- FEATURES ---------------- #

    # Eye openness
    left_eye_open = abs(lm[159].y - lm[145].y)
    right_eye_open = abs(lm[386].y - lm[374].y)
    avg_eye_open = (left_eye_open + right_eye_open) / 2

    # Depth (flat image detection)
    nose_depth = abs(lm[1].z)

    # ---------------- LIVENESS SCORE ---------------- #

    score = 0

    # Eye score (0–50)
    if avg_eye_open > 0.10:
        score += 50
    elif avg_eye_open > 0.05:
        score += 25

    # Depth score (0–50)
    if nose_depth > 0.15:
        score += 50
    elif nose_depth > 0.08:
        score += 25

    # ---------------- RESULT ---------------- #

    return f"LIVENESS SCORE: {score}%"

# ---------------- RUN ---------------- #
image_path = "images/1000063543.jpg"   
result = get_liveness_percentage(image_path)

print("\nRESULT:", result)