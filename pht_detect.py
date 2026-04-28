import cv2
import numpy as np
import mediapipe as mp
import time

# ---------------- FACE DETECTION ---------------- #
mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(min_detection_confidence=0.6)

# ---------------- LIVENESS FUNCTION ---------------- #
def liveness_score(face_roi):
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    noise = np.std(gray)

    edges = cv2.Canny(gray, 50, 150)
    edge_ratio = np.sum(edges) / (gray.shape[0] * gray.shape[1])

    score = (lap_var * 0.4) + (noise * 0.3) + (edge_ratio * 100)
    return score

# ---------------- SETTINGS ---------------- #
REAL_THRESHOLD = 1250
cap = cv2.VideoCapture(0)

start_time = time.time()
TIME_LIMIT = 10

#  store all scores here
all_scores = []

print("Running for 10 seconds...")

# ---------------- MAIN LOOP ---------------- #
while True:

    if time.time() - start_time > TIME_LIMIT:
        print("Time finished. Processing results...")
        break

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_detection.process(rgb)

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

                # store score
                all_scores.append(score)

                # optional live display
                label = "Detecting..."
                color = (255, 255, 0)

                # cv2.rectangle(frame, (x, y), (x+cw, y+ch), color, 2)
                center = (x + cw // 2, y + ch // 2)
                axes = (cw // 2, int(ch * 0.7))

                cv2.ellipse(frame, center, axes, 0, 0, 360, color, 2)
                cv2.putText(frame, label,
                            (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Anti-Spoofing System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Manual exit.")
        break

# ---------------- FINAL DECISION ---------------- #
cap.release()
cv2.destroyAllWindows()

if len(all_scores) > 0:
    avg_score = np.mean(all_scores)
    print(f"avg_score {avg_score}")

    print("\n======================")
    print("Average Score:", avg_score)

    if avg_score < REAL_THRESHOLD:
        print("FINAL RESULT: REAL FACE ")
    else:
        print("FINAL RESULT: FAKE / SPOOF ")
    print("======================")
else:
    print("No face detected.")