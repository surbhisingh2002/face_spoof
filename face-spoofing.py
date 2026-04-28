import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

# ---------------- FACE MESH ---------------- #
mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(min_detection_confidence=0.6)

# ---------------- SIMPLE CNN MODEL (placeholder logic) ---------------- #
# In real systems this is a trained anti-spoof model
# Here we simulate using texture + blur + motion cues


def liveness_score(face_roi):
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

    # 1. Blur detection (printed photos are often sharper or too smooth)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    # 2. Noise / texture check
    noise = np.std(gray)

    # 3. Edge density
    edges = cv2.Canny(gray, 50, 150)
    edge_ratio = np.sum(edges) / (gray.shape[0] * gray.shape[1])

    # Normalize score (simple heuristic)
    score = (lap_var * 0.4) + (noise * 0.3) + (edge_ratio * 100)

    return score

# thresholds (tune based on camera)
REAL_THRESHOLD = 1200

cap = cv2.VideoCapture(0)

frame_count = 0
motion_history = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_detection.process(rgb)

    label = "No Face"

    if results.detections:
        for det in results.detections:
            bbox = det.location_data.relative_bounding_box

            h, w, _ = frame.shape
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            cw = int(bbox.width * w)
            ch = int(bbox.height * h)

            x, y = max(0, x), max(0, y)

            face_roi = frame[y:y+ch, x:x+cw]

            if face_roi.size > 0:
                score = liveness_score(face_roi)
                print(score)
                # ---------------- CLASSIFICATION ---------------- #
                if score < REAL_THRESHOLD:
                    label = "REAL "
                    color = (0, 255, 0)
                else:
                    label = "FAKE / SPOOF "
                    color = (0, 0, 255)

                # draw box
                cv2.rectangle(frame, (x, y), (x+cw, y+ch), color, 2)

                # cv2.putText(frame, f"{label} | Score: {int(score)}",
                #             (x, y-10),
                cv2.putText(frame, f"{label}",
                            (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Anti-Spoofing System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()