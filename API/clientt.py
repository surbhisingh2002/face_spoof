import cv2
import requests
import os
import time
import mediapipe as mp

SERVER_URL = "http://localhost:8000/check_frame"
scores = []
TIME_LIMIT = 15  # seconds — change this as needed

# ── Create selfie folder ──
os.makedirs("selfie", exist_ok=True)

# ── MediaPipe setup ──
mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(min_detection_confidence=0.6)

cap = cv2.VideoCapture(0)
start_time = time.time()
verified = False  # track if liveness passed

print("Look at the camera... Press Q to quit")

while True:

    # ── Time limit check ──
    elapsed = time.time() - start_time
    remaining = int(TIME_LIMIT - elapsed)

    if elapsed > TIME_LIMIT:
        print("❌ Time limit reached! No real face detected.")
        print("No selfie saved.")
        break

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    h, w, _ = frame.shape
    display = frame.copy()

    # ── Oval settings ──
    center = (w // 2, h // 2)
    axes = (120, 160)

    # ── Send frame to server ──
    _, buffer = cv2.imencode('.jpg', frame)
    try:
        response = requests.post(
            SERVER_URL,
            files={"file": ("frame.jpg", buffer.tobytes(), "image/jpeg")},
            timeout=5
        )
        result = response.json()

    except Exception as e:
        print("Server error:", e)
        cv2.ellipse(display, center, axes, 0, 0, 360, (0, 0, 255), 2)
        cv2.putText(display, "Connecting to server...", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Liveness Check", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    status = result.get("status")

    # ── Face inside oval check ──
    face_inside = False
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    det_results = face_detector.process(rgb)

    if det_results.detections:
        det = det_results.detections[0]
        bbox = det.location_data.relative_bounding_box
        x = max(0, int(bbox.xmin * w))
        y = max(0, int(bbox.ymin * h))
        cw = int(bbox.width * w)
        ch = int(bbox.height * h)

        face_inside = (
            abs(x - center[0]) < axes[0] and
            abs(x + cw - center[0]) < axes[0] and
            abs(y - center[1]) < axes[1] and
            abs(y + ch - center[1]) < axes[1]
        )

    # ── Oval color ──
    oval_color = (0, 255, 0) if face_inside else (0, 0, 255)
    cv2.ellipse(display, center, axes, 0, 0, 360, oval_color, 3)

    # ── Corner guides ──
    tl = (center[0] - axes[0], center[1] - axes[1])
    br = (center[0] + axes[0], center[1] + axes[1])
    corner_len = 20
    cv2.line(display, tl, (tl[0] + corner_len, tl[1]), oval_color, 2)
    cv2.line(display, tl, (tl[0], tl[1] + corner_len), oval_color, 2)
    cv2.line(display, (br[0], tl[1]), (br[0] - corner_len, tl[1]), oval_color, 2)
    cv2.line(display, (br[0], tl[1]), (br[0], tl[1] + corner_len), oval_color, 2)
    cv2.line(display, (tl[0], br[1]), (tl[0] + corner_len, br[1]), oval_color, 2)
    cv2.line(display, (tl[0], br[1]), (tl[0], br[1] - corner_len), oval_color, 2)
    cv2.line(display, br, (br[0] - corner_len, br[1]), oval_color, 2)
    cv2.line(display, br, (br[0], br[1] - corner_len), oval_color, 2)

    # ── Timer on screen ──
    cv2.putText(display, f"Time left: {remaining}s", (w - 180, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # ── Status messages ──
    if status == "no_face":
        cv2.putText(display, "No face detected", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(display, "Put your face inside the oval", (30, 430),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    elif not face_inside:
        cv2.putText(display, "Align face inside oval", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    elif status == "fake":
        score = result.get("score", 0)
        cv2.putText(display, "FAKE / SPOOF DETECTED", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(display, f"Score: {score:.1f}", (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    elif status == "real":
        score = result.get("score", 0)
        scores.append(score)
        cv2.putText(display, f"REAL ({len(scores)}/5)", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(display, f"Score: {score:.1f}", (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display, "Hold still...", (30, 430),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # ── 5 REAL frames → save selfie ──
        if len(scores) >= 5:
            selfie_path = f"selfie/selfie_{int(time.time())}.jpg"
            cv2.imwrite(selfie_path, frame)
            verified = True
            print(f"✅ Liveness confirmed! Selfie saved → {selfie_path}")

            cv2.putText(display, "VERIFIED! Selfie Saved!", (30, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.imshow("Liveness Check", display)
            cv2.waitKey(2000)
            break

    cv2.imshow("Liveness Check", display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ── Final result ──
cap.release()
cv2.destroyAllWindows()

if verified:
    print("✅ Liveness PASSED — selfie saved in selfie/ folder")
else:
    print(" Liveness FAILED — no selfie saved")