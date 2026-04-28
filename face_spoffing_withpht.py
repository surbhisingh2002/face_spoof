import cv2
import numpy as np
import face_recognition
import mediapipe as mp

# ---------------- LOAD KNOWN FACE ---------------- #
known_image = face_recognition.load_image_file("images/1000063515.jpg")
known_encoding = face_recognition.face_encodings(known_image)[0]

# ---------------- FACE DETECTION (MediaPipe) ---------------- #
mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(min_detection_confidence=0.6)

# ---------------- LIVENESS FUNCTION ---------------- #
def liveness_score(face_roi):
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    noise = np.std(gray)

    edges = cv2.Canny(gray, 50, 150)
    edge_ratio = np.sum(edges) / (gray.shape[0] * gray.shape[1] + 1e-6)

    score = (lap_var * 0.4) + (noise * 0.3) + (edge_ratio * 100)

    return score

REAL_THRESHOLD = 1200

def check_liveness(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb)

    if not results.detections:
        return False

    det = results.detections[0]
    bbox = det.location_data.relative_bounding_box

    h, w, _ = frame.shape
    x = int(bbox.xmin * w)
    y = int(bbox.ymin * h)
    cw = int(bbox.width * w)
    ch = int(bbox.height * h)

    x, y = max(0, x), max(0, y)

    face_roi = frame[y:y+ch, x:x+cw]

    if face_roi.size == 0:
        return False

    score = liveness_score(face_roi)

    return score < REAL_THRESHOLD  # True = REAL

# ---------------- CAMERA ---------------- #
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ---------------- FACE MATCH ---------------- #
    faces = face_recognition.face_encodings(rgb)

    is_live = check_liveness(frame)

    status = "NO FACE"
    color = (0, 0, 255)

    if len(faces) > 0:
        match = face_recognition.compare_faces([known_encoding], faces[0])[0]

        if match and is_live:
            status = "ACCESS GRANTED ✔ (REAL HUMAN)"
            color = (0, 255, 0)

        elif match and not is_live:
            status = "SPOOF DETECTED "
            color = (0, 0, 255)

        else:
            status = "UNKNOWN USER "
            color = (0, 0, 255)

    cv2.putText(frame, status, (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Secure Authentication System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()