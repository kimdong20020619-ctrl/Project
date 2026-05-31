import os

os.makedirs('data', exist_ok=True)

files = {}

files['requirements.txt'] = """\
PySide6>=6.6.0
opencv-python>=4.8.0
mediapipe>=0.10.0
Pillow>=10.0.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
joblib>=1.3.0
matplotlib>=3.7.0
pyqtgraph>=0.13.0
gTTS>=2.4.0
pygame>=2.5.0
"""

files['00_test_setup.py'] = """\
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
    print(f"\\n[오류] pip install {' '.join(missing)}")
    sys.exit(1)

print("\\n모든 패키지 정상 설치됨!")
print("=" * 50)
print("웹캠 테스트 시작합니다. Q 키로 종료\\n")

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
"""

files['01_collect_data.py'] = """\
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import time
import platform
from PIL import ImageFont, ImageDraw, Image

_FONT_PATHS = {
    'Windows': 'C:/Windows/Fonts/malgun.ttf',
    'Darwin':  '/System/Library/Fonts/AppleGothic.ttf',
    'Linux':   '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
}

def _kr_font(size):
    path = _FONT_PATHS.get(platform.system(), '')
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()

FONT_LG = _kr_font(36)
FONT_MD = _kr_font(24)
FONT_SM = _kr_font(18)

def put_kr(frame, text, xy, font, color=(255, 255, 255)):
    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(pil).text(xy, text, font=font, fill=color)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

SIGNS = [
    'ㄱ','ㄴ','ㄷ','ㄹ','ㅁ','ㅂ','ㅅ','ㅇ',
    'ㅈ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ',
    'ㅏ','ㅑ','ㅓ','ㅕ','ㅗ','ㅛ','ㅜ','ㅠ','ㅡ','ㅣ',
]
SAMPLES_PER_SIGN = 100
CAPTURE_INTERVAL = 0.05
OUTPUT_PATH      = 'data/landmarks_raw.csv'

def normalize(hand_landmarks):
    pts = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark], dtype=np.float32)
    pts -= pts[0]
    scale = float(np.linalg.norm(pts[9]))
    if scale > 1e-6:
        pts /= scale
    return pts.flatten().tolist()

def collect():
    os.makedirs('data', exist_ok=True)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[오류] 웹캠을 찾을 수 없습니다.")
        return

    mp_hands  = mp.solutions.hands
    mp_draw   = mp.solutions.drawing_utils
    DOT       = mp_draw.DrawingSpec(color=(60,  220, 60), thickness=4)
    LINE      = mp_draw.DrawingSpec(color=(0,   160,  0), thickness=2)

    all_rows  = []
    quit_flag = False

    print(f"\\n수어 목록 ({len(SIGNS)}개): {' '.join(SIGNS)}")
    print(f"수어 1개당 {SAMPLES_PER_SIGN}개 | 총 목표 {len(SIGNS)*SAMPLES_PER_SIGN}개")
    print("조작: SPACE=시작  S=건너뛰기  Q=종료\\n")

    with mp_hands.Hands(max_num_hands=1,
                        min_detection_confidence=0.75,
                        min_tracking_confidence=0.6) as hands:

        for idx, sign in enumerate(SIGNS):
            if quit_flag:
                break

            count      = 0
            collecting = False
            last_t     = 0.0

            print(f"  [{idx+1:02d}/{len(SIGNS)}] '{sign}'  SPACE:시작  S:건너뜀  Q:종료")

            while count < SAMPLES_PER_SIGN:
                ret, frame = cap.read()
                if not ret:
                    break

                frame    = cv2.flip(frame, 1)
                h, w     = frame.shape[:2]
                rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result   = hands.process(rgb)
                detected = result.multi_hand_landmarks is not None
                now      = time.time()

                if detected:
                    for hl in result.multi_hand_landmarks:
                        mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS, DOT, LINE)

                if collecting and detected and (now - last_t) >= CAPTURE_INTERVAL:
                    for hl in result.multi_hand_landmarks:
                        if count < SAMPLES_PER_SIGN:
                            all_rows.append([sign] + normalize(hl))
                            count += 1
                    last_t = now

                ov = frame.copy()
                cv2.rectangle(ov, (0, 0), (w, 130), (0, 0, 0), -1)
                cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)

                frame = put_kr(frame,
                    f"[{idx+1}/{len(SIGNS)}]  '{sign}'  수어를 카메라에 보여주세요",
                    (12, 6), FONT_LG, (255, 255, 255))

                if not collecting:
                    frame = put_kr(frame,
                        "SPACE: 수집 시작   S: 건너뛰기   Q: 종료",
                        (12, 56), FONT_SM, (160, 160, 160))
                else:
                    prog_w = int((count / SAMPLES_PER_SIGN) * (w - 28))
                    cv2.rectangle(frame, (14, 92), (w-14, 118), (40, 40, 40), -1)
                    cv2.rectangle(frame, (14, 92), (14+prog_w, 118),
                                  (30, 200, 90) if detected else (80, 80, 210), -1)
                    frame = put_kr(frame,
                        f"수집 중...  {count} / {SAMPLES_PER_SIGN}"
                        + ("" if detected else "  ← 손이 보이지 않습니다"),
                        (12, 56), FONT_MD,
                        (30, 215, 100) if detected else (80, 100, 255))

                cv2.imshow('수어 데이터 수집기  |  SPACE 시작  S 건너뛰기  Q 종료', frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), ord('Q')):
                    quit_flag = True
                    break
                if key in (ord('s'), ord('S')):
                    print(f"    건너뜀")
                    break
                if key == ord(' ') and not collecting:
                    collecting = True
                    print(f"    수집 시작!")

            if not quit_flag and count >= SAMPLES_PER_SIGN:
                print(f"    '{sign}' 완료 ({count}개)")

    cap.release()
    cv2.destroyAllWindows()

    if not all_rows:
        print("\\n수집된 데이터가 없습니다.")
        return

    cols = ['label'] + [f'{ax}{i}' for i in range(21) for ax in ('x', 'y')]
    df   = pd.DataFrame(all_rows, columns=cols)
    df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')

    print(f"\\n저장 완료: {OUTPUT_PATH}")
    print(f"총 {len(df)}개 샘플 | {df['label'].nunique()}개 클래스")
    for s, c in df['label'].value_counts().sort_index().items():
        print(f"  {s}  {'█'*(c//5)} {c}")
    print("\\n다음 단계: python 02_train_model.py")

if __name__ == '__main__':
    collect()
"""

for filename, content in files.items():
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"[OK] {filename} 생성됨")

print("\n모든 파일 생성 완료!")
print("다음 명령어를 실행하세요:")
print("  pip install -r requirements.txt")
print("  python 00_test_setup.py")
