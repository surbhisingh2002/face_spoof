import cv2
import mediapipe as mp
import time
import numpy as np

# ---------------- FACE DETECTION ---------------- #
mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(min_detection_confidence=0.6)

cap = cv2.VideoCapture(0)

capture_start = None
selfie_taken = False

TIME_LIMIT = 10
start_time = time.time()

print("Align face inside oval...")

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

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb)

    display_frame = frame.copy()

    center = (w // 2, h // 2)
    axes = (110, 150)

    face_inside = False

    if results.detections:
        for det in results.detections:

            bbox = det.location_data.relative_bounding_box

            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            cw = int(bbox.width * w)
            ch = int(bbox.height * h)
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

                cv2.putText(display_frame, "Good Position",
                            (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 255, 0), 2)  # green text

    # ---------------- COLOR CHANGE ADDED HERE ---------------- #
    if face_inside:
        color = (0, 255, 0)   # GREEN
    else:
        color = (0, 0, 255)   # RED

    cv2.ellipse(display_frame, center, axes, 0, 0, 360, color, 2)

    # ---------------- TIMER LOGIC ---------------- #
    if face_inside and not selfie_taken:
        
        if capture_start is None:
            capture_start = time.time()

        elif time.time() - capture_start >= 3:
            cv2.imwrite("selfie.jpg", clean_frame)
            print(" Clean selfie captured (NO OVAL)")
            selfie_taken = True
            

    else:
        capture_start = None

    cv2.putText(display_frame, "Put face inside oval",
                (50, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0), 2)

    cv2.imshow("Oval Selfie Capture", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()