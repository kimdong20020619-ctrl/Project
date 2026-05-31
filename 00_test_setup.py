import sys

print("=" * 50)
print("  패키지 설치 확인 중...")
print("=" * 50)

REQUIRED = {
    "cv2":       "opencv-python",
    "mediapipe": "mediapipe",
    "numpy":     "numpy",
    "pandas":    "pandas",
    "sklearn":   "scikit-learn",
    "PIL":       "Pillow",
    "PySide6":   "PySide6",
}

missing = []
for mod, pkg in REQUIRED.items():
    try:
        m = __import__(mod)
        ver = getattr(m, "__version__", "?")
        print(f"  [OK] {pkg:<22} {ver}")
    except ImportError:
        print(f"  [!!] {pkg:<22} 미설치")
        missing.append(pkg)

if missing:
    print(f"\n[오류] pip install {' '.join(missing)}")
    sys.exit(1)

print("\n모든 패키지 정상 설치됨!")
print("=" * 50)
print("웹캠 테스트 시작합니다. Q 키로 종료\n")

import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[오류] 웹캠을 찾을 수 없습니다.")
    sys.exit(1)

mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
prev_t   = time.time()

with mp_hands.Hands(max_num_hands=1,
                    min_detection_confidence=0.7,
                    min_tracking_confidence=0.5) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame  = cv2.flip(frame, 1)
        h, w   = frame.shape[:2]
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        now    = time.time()
        fps    = 1.0 / max(now - prev_t, 1e-6)
        prev_t = now

        if result.multi_hand_landmarks:
            for hl in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame, hl, mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(60, 230, 60), thickness=3),
                    mp_draw.DrawingSpec(color=(0,  180,  0), thickness=2),
                )
            cv2.putText(frame, "HAND DETECTED - 21 landmarks OK",
                       (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 230, 60), 2)
        else:
            cv2.putText(frame, "Show your hand to the camera",
                       (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 255), 2)

        cv2.putText(frame, f"FPS: {fps:.0f}",
                   (w - 90, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(frame, "Press Q to quit",
                   (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        cv2.imshow("Setup Test - Sign Language App", frame)
        if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
            break

cap.release()
cv2.destroyAllWindows()
print("테스트 완료! 다음: python 01_collect_data.py")
