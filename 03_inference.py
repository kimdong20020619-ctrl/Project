import cv2
import json
import pickle
import platform
import urllib.request
from collections import deque
from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker, HandLandmarkerOptions, RunningMode,
)
from PIL import ImageFont, ImageDraw, Image

MODEL_DIR     = Path('model')
MODEL_PATH    = MODEL_DIR / 'sign_model.pkl'
ENCODER_PATH  = MODEL_DIR / 'label_encoder.pkl'
META_PATH     = MODEL_DIR / 'model_meta.json'
LSTM_PATH     = MODEL_DIR / 'sign_lstm_state.pt'
HAND_TASK     = Path('hand_landmarker.task')   # MediaPipe 0.10+ 모델 파일

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

def put_kr(frame, text, xy, font, color=(255, 255, 255)):
    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(pil).text(xy, text, font=font, fill=color)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

FONT_LG = _kr_font(44)
FONT_MD = _kr_font(28)
FONT_SM = _kr_font(18)

# MediaPipe 손 연결선 (랜드마크 그리기용)
_HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]


def _ensure_hand_task():
    """hand_landmarker.task 파일이 없으면 다운로드."""
    if HAND_TASK.exists():
        return
    url = (
        'https://storage.googleapis.com/mediapipe-models/'
        'hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
    )
    print(f'hand_landmarker.task 다운로드 중... ({url})')
    urllib.request.urlretrieve(url, HAND_TASK)
    print('다운로드 완료')


def _normalize_pts(pts: np.ndarray) -> np.ndarray:
    pts = pts.copy()
    pts -= pts[0]
    scale = float(np.linalg.norm(pts[9]))
    if scale > 1e-6:
        pts /= scale
    return pts


class SignInferencer:
    def __init__(self):
        for p in (MODEL_PATH, ENCODER_PATH):
            if not p.exists():
                raise FileNotFoundError(
                    f'모델 파일 없음: {p}\n'
                    'Colab 학습 후 model/ 폴더에 파일을 넣고 재실행하세요.'
                )

        with open(ENCODER_PATH, 'rb') as f:
            self.encoder = pickle.load(f)

        meta = {}
        if META_PATH.exists():
            with open(META_PATH, encoding='utf-8') as f:
                meta = json.load(f)

        self.seq_len     = meta.get('seq_len', 32)
        self.feature_dim = meta.get('feature_dim', 42)
        self.model_type  = meta.get('best_model', 'sklearn')
        self._buffer     = deque(maxlen=self.seq_len)

        with open(MODEL_PATH, 'rb') as f:
            saved = pickle.load(f)

        if isinstance(saved, dict) and saved.get('type') == 'lstm':
            self._load_lstm(saved)
        else:
            self._model   = saved.get('model', saved) if isinstance(saved, dict) else saved
            self._is_lstm = False

        # MediaPipe HandLandmarker (Tasks API, 0.10+)
        _ensure_hand_task()
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(HAND_TASK)),
            running_mode=RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.75,
            min_hand_presence_confidence=0.6,
        )
        self._landmarker  = HandLandmarker.create_from_options(options)
        self._last_result = None

        print(f'모델 로드 완료 | 타입: {self.model_type} | 단어: {len(self.encoder.classes_)}개')

    def _load_lstm(self, saved: dict):
        import torch
        from torch import nn

        feature_dim = self.feature_dim
        num_classes = len(self.encoder.classes_)

        class _LSTM(nn.Module):
            def __init__(self, input_dim, hidden=128, layers=2, n_cls=30, dropout=0.3):
                super().__init__()
                self.lstm = nn.LSTM(input_dim, hidden, layers, batch_first=True, dropout=dropout)
                self.head = nn.Sequential(
                    nn.Linear(hidden, 128), nn.ReLU(), nn.Dropout(dropout),
                    nn.Linear(128, n_cls),
                )
            def forward(self, x):
                out, _ = self.lstm(x)
                return self.head(out[:, -1])

        device = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
        state_path = MODEL_DIR / saved.get('state_path', 'sign_lstm_state.pt')
        lstm = _LSTM(feature_dim, n_cls=num_classes)
        lstm.load_state_dict(__import__('torch').load(state_path, map_location=device))
        lstm.eval()
        self._model   = lstm.to(device)
        self._device  = device
        self._is_lstm = True

    def _detect(self, frame_bgr):
        """BGR 프레임 → MediaPipe 감지 결과."""
        rgb      = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result   = self._landmarker.detect(mp_image)
        self._last_result = result
        return result

    def _keypoints_from_result(self, result) -> np.ndarray | None:
        if not result.hand_landmarks:
            return None
        lms = result.hand_landmarks[0]   # NormalizedLandmark 리스트
        pts = np.array([[lm.x, lm.y] for lm in lms], dtype=np.float32)

        # 왼손이면 x 반전 (오른손 기준 통일)
        if result.handedness and result.handedness[0]:
            label = result.handedness[0][0].category_name  # 'Left' or 'Right'
            if label == 'Left':
                pts[:, 0] *= -1

        return _normalize_pts(pts).flatten()

    def _buffer_to_seq(self) -> np.ndarray:
        buf = list(self._buffer)
        if len(buf) < self.seq_len:
            pad = [np.zeros(self.feature_dim, dtype=np.float32)] * (self.seq_len - len(buf))
            buf = pad + buf
        return np.stack(buf, axis=0)  # (seq_len, feature_dim)

    def predict(self, frame_bgr):
        """BGR 프레임 → (단어, 신뢰도, top3). 손 없으면 (None, 0.0, [])."""
        result = self._detect(frame_bgr)
        vec    = self._keypoints_from_result(result)

        if vec is None:
            return None, 0.0, []

        self._buffer.append(vec)

        if len(self._buffer) < max(1, self.seq_len // 2):
            return None, 0.0, []

        seq = self._buffer_to_seq()

        if self._is_lstm:
            import torch
            with torch.no_grad():
                xb    = torch.tensor(seq[None], dtype=torch.float32, device=self._device)
                logit = self._model(xb)
                proba = torch.softmax(logit, dim=1).cpu().numpy()[0]
        else:
            flat  = np.concatenate([seq.mean(axis=0), seq.std(axis=0)]).reshape(1, -1)
            proba = self._model.predict_proba(flat)[0]

        top3_idx = proba.argsort()[-3:][::-1]
        top3 = [(self.encoder.classes_[i], float(proba[i])) for i in top3_idx]
        return top3[0][0], top3[0][1], top3

    def draw_landmarks(self, frame, *_):
        """손 랜드마크를 프레임에 그린다. 두 번째 인수(draw_utils)는 무시."""
        result = self._last_result
        if result is None or not result.hand_landmarks:
            return frame
        h, w = frame.shape[:2]
        for lms in result.hand_landmarks:
            pts_px = [(int(lm.x * w), int(lm.y * h)) for lm in lms]
            for s, e in _HAND_CONNECTIONS:
                cv2.line(frame, pts_px[s], pts_px[e], (0, 160, 0), 2)
            for px in pts_px:
                cv2.circle(frame, px, 4, (60, 220, 60), -1)
        return frame

    def close(self):
        self._landmarker.close()


def run_realtime():
    inferencer = SignInferencer()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('[오류] 웹캠을 찾을 수 없습니다.')
        return

    BUFFER_SIZE = 10
    word_buffer = []
    stable_word = ''
    stable_conf = 0.0
    sentence    = []

    print('실시간 수어 인식 | SPACE: 단어 추가  C: 문장 지우기  Q: 종료')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]
        frame = inferencer.draw_landmarks(frame)

        word, conf, top3 = inferencer.predict(frame)

        if word:
            word_buffer.append(word)
        if len(word_buffer) > BUFFER_SIZE:
            word_buffer.pop(0)
        if word_buffer:
            stable_word = max(set(word_buffer), key=word_buffer.count)
            stable_conf = conf

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 130), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        if stable_word:
            color = (30, 215, 100) if stable_conf >= 0.7 else (80, 130, 255)
            frame = put_kr(frame, f'인식: {stable_word}', (14, 8), FONT_LG, color)
            frame = put_kr(frame, f'신뢰도 {stable_conf*100:.1f}%', (14, 65), FONT_MD, (200, 200, 200))
        else:
            frame = put_kr(frame, '손을 카메라에 보여주세요', (14, 8), FONT_LG, (120, 120, 120))

        if top3:
            for i, (wl, p) in enumerate(top3):
                frame = put_kr(frame, f'{i+1}. {wl} ({p*100:.0f}%)',
                               (w - 240, h - 95 + i * 30), FONT_SM, (200, 200, 200))

        if sentence:
            frame = put_kr(frame, '  '.join(sentence), (14, h - 40), FONT_SM, (255, 230, 100))

        frame = put_kr(frame, 'SPACE: 추가  C: 지우기  Q: 종료', (14, h - 68), FONT_SM, (120, 120, 120))
        cv2.imshow('한국수어 인식', frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q')):
            break
        if key == ord(' ') and stable_word:
            sentence.append(stable_word)
            word_buffer.clear()
            stable_word = ''
        if key in (ord('c'), ord('C')):
            sentence.clear()

    cap.release()
    cv2.destroyAllWindows()
    inferencer.close()


if __name__ == '__main__':
    run_realtime()
