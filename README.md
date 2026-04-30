# Face Liveness Detection

Face anti-spoofing system using MediaPipe + OpenCV. Detects if a face is **REAL** or **FAKE**.

---

## Files

| File | What it does |
|---|---|
| `face-spoofing.py` | Runs camera for 10 seconds, gives REAL or FAKE at the end |
| `pht_detect.py` | Opens camera with oval UI, saves selfie when face inside oval |
| `pht_liveliness.py` | Oval UI + liveness check combined, saves selfie if REAL |
---

## Install
```bash
pip install opencv-python mediapipe numpy face-recognition
```

---

## Run any file
```bash
python face-spoofing.py
python pht_detect.py
python pht_liveliness.py
python face_spoffing_withpht.py
```

---

## How liveness works
```
Captures face → checks sharpness + noise + edges → gives score
Score above 500 → REAL ✅ 
Score below 500 → FAKE ❌
```

---

## Notes
- `face_spoffing_withpht.py` needs a known face image at `images/your_photo.jpg`
- `pht_liveliness.py` saves selfie in `selfie/` folder
- Press **Q** to quit any script


- less then 500 is fake based on pht captured by front camera of a pht from another phn. and more than this is fake