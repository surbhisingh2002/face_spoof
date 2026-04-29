import cv2
import mediapipe as mp
import time
import numpy as np

# ---------------- LIVENESS FUNCTION ---------------- #
def liveness_score(face_roi):
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    noise = np.std(gray)

    edges = cv2.Canny(gray, 50, 150)
    edge_ratio = np.sum(edges) / (gray.shape[0] * gray.shape[1])

    score = (lap_var * 0.4) + (noise * 0.3) + (edge_ratio * 100)
    return score

# ---------------- FACE DETECTION ---------------- #
mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(min_detection_confidence=0.6)

cap = cv2.VideoCapture(0)

capture_start = None
selfie_taken = False

all_scores = []
REAL_THRESHOLD = 1400

TIME_LIMIT = 10
start_time = time.time()

print("Align face inside oval...")

# ---------------- MAIN LOOP ---------------- #
while True:

    if time.time() - start_time > TIME_LIMIT:
        print("Time finished.")
        break

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    h, w, _ = frame.shape

    clean_frame = frame.copy()
    display_frame = frame.copy()

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb)

    center = (w // 2, h // 2)
    axes = (110, 150)

    face_inside = False
    face_roi = None

    # ---------------- FACE DETECTION ---------------- #
    if results.detections:
        for det in results.detections:

            bbox = det.location_data.relative_bounding_box

            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            cw = int(bbox.width * w)
            ch = int(bbox.height * h)

            x, y = max(0, x), max(0, y)
            left = x
            right = x + cw
            top = y
            bottom = y + ch
            face_center = (x + cw // 2, y + ch // 2)

            # check inside oval
            if (abs(left - center[0]) < axes[0] and
                abs(right - center[0]) < axes[0] and
                abs(top - center[1]) < axes[1] and
                abs(bottom - center[1]) < axes[1]):

                face_inside = True

                face_roi = frame[y:y+ch, x:x+cw]

                cv2.putText(display_frame, "Good Position",
                            (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 255, 0), 2)

    # ---------------- OVAL COLOR ---------------- #
    if face_inside:
        color = (0, 255, 0)   # GREEN
    else:
        color = (0, 0, 255)   # RED

    cv2.ellipse(display_frame, center, axes, 0, 0, 360, color, 2)

    # ---------------- LIVENESS ---------------- #
    if face_roi is not None and face_roi.size > 0:
        score = liveness_score(face_roi)
        all_scores.append(score)

        cv2.putText(display_frame, "Detecting...",
                    (50, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)

    # ---------------- SELFIE TIMER ---------------- #
    if face_inside and not selfie_taken:

        if capture_start is None:
            capture_start = time.time()

        elif time.time() - capture_start >= 3:
            cv2.imwrite("selfie/selfie.jpg", clean_frame)
            print("Clean selfie captured")
            selfie_taken = True

    else:
        capture_start = None

    # ---------------- UI TEXT ---------------- #
    cv2.putText(display_frame, "Put face inside oval",
                (50, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0), 2)

    cv2.imshow("Oval Selfie Capture", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ---------------- FINAL RESULT ---------------- #
cap.release()
cv2.destroyAllWindows()

if len(all_scores) > 0:
    avg_score = np.mean(all_scores)

    print("\n======================")
    print("Average Score:", avg_score)

    if avg_score < REAL_THRESHOLD:
        print("FINAL RESULT: REAL FACE")
    else:
        print("FINAL RESULT: FAKE / SPOOF")
    print("======================")

else:
    print("No face detected.")