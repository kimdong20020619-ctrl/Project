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
    # 인사/기본
    '안녕하세요', '감사합니다', '죄송합니다', '괜찮아요', '잠깐만요',
    # 감정
    '좋아요', '싫어요', '기뻐요', '슬퍼요', '화나요',
    # 의사표현
    '네', '아니요', '모르겠어요', '도와주세요', '아파요',
    # 일상명사
    '밥', '물', '화장실', '집', '병원',
    # 사람
    '나', '너', '엄마', '아빠', '친구',
    # 동작 (움직임 포함)
    '먹다', '마시다', '가다', '오다', '자다',
]
SAMPLES_PER_SIGN = 150
CAPTURE_INTERVAL = 0.05
OUTPUT_PATH      = 'data/landmarks_raw.csv'

def normalize(hand_landmarks, handedness) -> list:
    pts = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark], dtype=np.float32)
    pts -= pts[0]
    scale = float(np.linalg.norm(pts[9]))
    if scale > 1e-6:
        pts /= scale
    if handedness.classification[0].label == 'Left':
        pts[:, 0] *= -1
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

    print(f"\n수어 목록 ({len(SIGNS)}개): {' '.join(SIGNS)}")
    print(f"수어 1개당 {SAMPLES_PER_SIGN}개 | 총 목표 {len(SIGNS)*SAMPLES_PER_SIGN}개")
    print("조작: SPACE=시작  S=건너뛰기  Q=종료\n")

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
                    for i, hl in enumerate(result.multi_hand_landmarks):
                         if count < SAMPLES_PER_SIGN:
                            handedness = result.multi_handedness[i]
                            all_rows.append([sign] + normalize(hl, handedness))
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
        print("\n수집된 데이터가 없습니다.")
        return

    cols = ['label'] + [f'{ax}{i}' for i in range(21) for ax in ('x', 'y')]
    df   = pd.DataFrame(all_rows, columns=cols)
    df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')

    print(f"\n저장 완료: {OUTPUT_PATH}")
    print(f"총 {len(df)}개 샘플 | {df['label'].nunique()}개 클래스")
    for s, c in df['label'].value_counts().sort_index().items():
        print(f"  {s}  {'█'*(c//5)} {c}")
    print("\n다음 단계: python 02_train_model.py")

if __name__ == '__main__':
    collect()
