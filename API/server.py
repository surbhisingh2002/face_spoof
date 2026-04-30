from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np
import mediapipe as mp

app = FastAPI()

mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(min_detection_confidence=0.6)

def liveness_score(face_roi):
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    noise = np.std(gray)
    edges = cv2.Canny(gray, 50, 150)
    edge_ratio = np.sum(edges) / (gray.shape[0] * gray.shape[1])
    return (lap_var * 0.4) + (noise * 0.3) + (edge_ratio * 100)

@app.post("/check_frame")
async def check_frame(file: UploadFile = File(...)):

    # ── 1. Decode the uploaded image ──
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        return {"status": "error", "message": "Invalid image"}

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb)

    # ── 2. Detect face ──
    if not results.detections:
        return {"status": "no_face", "message": "No face detected"}

    # ── 3. Only one face allowed ──
    if len(results.detections) > 1:
        return {"status": "error", "message": "Multiple faces detected. Only one face allowed."}

    det = results.detections[0]       # use first and only face
    bbox = det.location_data.relative_bounding_box
    x = max(0, int(bbox.xmin * w))
    y = max(0, int(bbox.ymin * h))
    cw = int(bbox.width * w)
    ch = int(bbox.height * h)
    face_roi = frame[y:y+ch, x:x+cw]

    if face_roi.size == 0:
        return {"status": "no_face", "message": "Face ROI empty"}

    # ── 4. Check liveness ──
    REAL_THRESHOLD = 500
    score = liveness_score(face_roi)

    if score <= REAL_THRESHOLD:
        return {
            "status": "fake",
            "score": float(score),
            "message": "Spoof detected. Try again."
        }

    # ── 5. Save selfie and return ──
    cv2.imwrite("selfie.jpg", frame)
    _, buffer = cv2.imencode('.jpg', frame)

    return {
        "status": "real",
        "score": float(score),
        "selfie": buffer.tobytes().hex()
    }