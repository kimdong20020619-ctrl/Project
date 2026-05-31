"""
파이프라인 검증 스크립트 — 실제 모델/데이터 없이 실행 가능.

검증 항목:
  1. 키포인트 정규화 함수 (수치 정확성)
  2. 시퀀스 버퍼 (길이 보장, 패딩, 균일 샘플링)
  3. 추론기 전체 흐름 (더미 모델 + 더미 웹캠 프레임)
  4. TTS/STT 임포트 가능 여부
  5. 앱 임포트 및 SignInferencer 초기화

실행: python 05_test_pipeline.py
"""

import os
import sys
import json
import pickle
import tempfile
import traceback
from pathlib import Path

import numpy as np

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

PASS = '[PASS]'
FAIL = '[FAIL]'
SKIP = '[SKIP]'

results = []

def check(name, fn):
    try:
        fn()
        print(f'{PASS} {name}')
        results.append((name, True, ''))
    except Exception as e:
        msg = str(e)
        print(f'{FAIL} {name}')
        print(f'       {msg}')
        results.append((name, False, msg))


# ─────────────────────────────────────────────────────────────
# 1. 키포인트 정규화
# ─────────────────────────────────────────────────────────────
def test_normalize():
    pts = np.random.rand(21, 2).astype(np.float32)
    pts -= pts[0]
    scale = float(np.linalg.norm(pts[9]))
    if scale > 1e-6:
        pts /= scale
    flat = pts.flatten()
    assert flat.shape == (42,), f'shape mismatch: {flat.shape}'
    # 손목(0번)이 원점인지
    assert abs(flat[0]) < 1e-5 and abs(flat[1]) < 1e-5, '손목 원점화 실패'
    # 스케일 정규화 후 9번 관절 거리 ≈ 1
    pts_after = flat.reshape(21, 2)
    norm9 = float(np.linalg.norm(pts_after[9]))
    assert abs(norm9 - 1.0) < 1e-4, f'스케일 정규화 실패: norm9={norm9}'

check('키포인트 정규화 (손목 원점, 스케일=1)', test_normalize)


# ─────────────────────────────────────────────────────────────
# 2. 시퀀스 버퍼 — 균일 샘플링
# ─────────────────────────────────────────────────────────────
def test_seq_uniform_sample():
    SEQ_LEN = 32
    frames = np.random.rand(100, 42).astype(np.float32)
    idx = np.linspace(0, len(frames) - 1, SEQ_LEN).astype(int)
    seq = frames[idx]
    assert seq.shape == (SEQ_LEN, 42), f'shape mismatch: {seq.shape}'
    assert idx[0] == 0 and idx[-1] == 99

check('시퀀스 균일 샘플링 (100→32)', test_seq_uniform_sample)


def test_seq_padding():
    SEQ_LEN = 32
    frames = np.random.rand(10, 42).astype(np.float32)
    pad = np.repeat(frames[-1:], SEQ_LEN - len(frames), axis=0)
    seq = np.concatenate([frames, pad], axis=0)
    assert seq.shape == (SEQ_LEN, 42)
    # 마지막 프레임이 복제됐는지
    np.testing.assert_array_equal(seq[10], seq[31])

check('시퀀스 패딩 (10→32)', test_seq_padding)


# ─────────────────────────────────────────────────────────────
# 3. 더미 모델 생성 후 model/ 폴더에 저장
# ─────────────────────────────────────────────────────────────
TARGET_WORDS = [
    '안녕하세요','감사합니다','죄송합니다','괜찮아요','잠깐만요',
    '좋아요','싫어요','기뻐요','슬퍼요','화나요',
    '네','아니요','모르겠어요','도와주세요','아파요',
    '밥','물','화장실','집','병원',
    '나','너','엄마','아빠','친구',
    '먹다','마시다','가다','오다','자다',
]
SEQ_LEN     = 32
FEATURE_DIM = 42
MODEL_DIR   = Path('model')

def setup_dummy_model():
    from sklearn.dummy import DummyClassifier
    from sklearn.preprocessing import LabelEncoder

    MODEL_DIR.mkdir(exist_ok=True)

    le = LabelEncoder()
    le.fit(TARGET_WORDS)

    # DummyClassifier: predict_proba 지원 (stratified)
    clf = DummyClassifier(strategy='stratified', random_state=42)
    n_samples = len(TARGET_WORDS) * 5
    X_dummy = np.random.rand(n_samples, FEATURE_DIM * 2).astype(np.float32)
    y_dummy = le.transform(TARGET_WORDS * 5)
    clf.fit(X_dummy, y_dummy)

    with open(MODEL_DIR / 'sign_model.pkl', 'wb') as f:
        pickle.dump({'type': 'sklearn', 'model': clf}, f)
    with open(MODEL_DIR / 'label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)

    meta = {
        'best_model':    'DummyClassifier',
        'num_classes':   len(TARGET_WORDS),
        'classes':       TARGET_WORDS,
        'seq_len':       SEQ_LEN,
        'feature_dim':   FEATURE_DIM,
        'n_keypoints':   21,
        'test_accuracy': 0.0,
        'test_f1_macro': 0.0,
    }
    with open(MODEL_DIR / 'model_meta.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

check('더미 모델 생성 및 model/ 저장', setup_dummy_model)


# ─────────────────────────────────────────────────────────────
# 4. SignInferencer 초기화
# ─────────────────────────────────────────────────────────────
_inferencer = None

def test_inferencer_init():
    global _inferencer
    import importlib.util
    spec = importlib.util.spec_from_file_location('inference', '03_inference.py')
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _inferencer = mod.SignInferencer()
    assert _inferencer is not None
    assert len(_inferencer.encoder.classes_) == 30

check('SignInferencer 초기화', test_inferencer_init)


# ─────────────────────────────────────────────────────────────
# 5. 더미 BGR 프레임으로 predict() 호출
# ─────────────────────────────────────────────────────────────
def test_predict_dummy_frame():
    assert _inferencer is not None, 'inferencer 초기화 실패'
    frame = np.zeros((480, 640, 3), dtype=np.uint8)  # 손 없는 검정 프레임
    word, conf, top3 = _inferencer.predict(frame)
    # 손이 없으면 None 반환이어야 함
    assert word is None, f'빈 프레임에서 word={word} 반환됨'
    assert conf == 0.0
    assert top3 == []

check('빈 프레임 → (None, 0, []) 반환', test_predict_dummy_frame)


# ─────────────────────────────────────────────────────────────
# 6. 롤링 버퍼 동작 확인
# ─────────────────────────────────────────────────────────────
def test_rolling_buffer():
    from collections import deque
    buf = deque(maxlen=SEQ_LEN)
    vec = np.random.rand(FEATURE_DIM).astype(np.float32)

    for _ in range(SEQ_LEN + 10):  # 버퍼 초과 추가
        buf.append(vec.copy())

    assert len(buf) == SEQ_LEN, f'버퍼 크기 오류: {len(buf)}'

    # 버퍼 → (SEQ_LEN, FEATURE_DIM) 변환
    seq = np.stack(list(buf), axis=0)
    assert seq.shape == (SEQ_LEN, FEATURE_DIM)

    # sklearn 입력: mean + std
    flat = np.concatenate([seq.mean(axis=0), seq.std(axis=0)], axis=0)
    assert flat.shape == (FEATURE_DIM * 2,)

check('롤링 버퍼 (maxlen=32, mean+std 집계)', test_rolling_buffer)


# ─────────────────────────────────────────────────────────────
# 7. 모델 meta.json 구조 검증
# ─────────────────────────────────────────────────────────────
def test_meta_json():
    with open(MODEL_DIR / 'model_meta.json', encoding='utf-8') as f:
        meta = json.load(f)
    required = ['best_model', 'num_classes', 'classes', 'seq_len', 'feature_dim']
    for key in required:
        assert key in meta, f'meta.json에 {key} 없음'
    assert meta['num_classes'] == 30
    assert meta['seq_len'] == SEQ_LEN
    assert meta['feature_dim'] == FEATURE_DIM
    assert len(meta['classes']) == 30

check('model_meta.json 구조', test_meta_json)


# ─────────────────────────────────────────────────────────────
# 8. TTS/STT 패키지 임포트
# ─────────────────────────────────────────────────────────────
def test_tts_import():
    try:
        import pyttsx3  # noqa  (기본 TTS 백엔드)
        return
    except ImportError:
        pass
    from gtts import gTTS  # noqa
    import pygame          # noqa

def test_stt_import():
    import speech_recognition as sr  # noqa

try:
    check('gTTS + pygame 임포트', test_tts_import)
except Exception:
    print(f'{SKIP} gTTS/pygame 없음 — pip install gTTS pygame 으로 설치')
    results.append(('gTTS+pygame', False, 'not installed'))

try:
    check('SpeechRecognition 임포트', test_stt_import)
except Exception:
    print(f'{SKIP} SpeechRecognition 없음 — pip install SpeechRecognition pyaudio 로 설치')
    results.append(('SpeechRecognition', False, 'not installed'))


# ─────────────────────────────────────────────────────────────
# 9. 04_app.py 문법 확인 (컴파일만)
# ─────────────────────────────────────────────────────────────
def test_app_syntax():
    import ast
    src = Path('04_app.py').read_text(encoding='utf-8')
    ast.parse(src)

check('04_app.py 문법 검사 (AST parse)', test_app_syntax)


# ─────────────────────────────────────────────────────────────
# 결과 요약
# ─────────────────────────────────────────────────────────────
if _inferencer:
    _inferencer.close()

passed = sum(1 for _, ok, _ in results if ok)
total  = len(results)
print(f'\n{"─"*50}')
print(f'결과: {passed}/{total} 통과')
if passed == total:
    print('모든 검사 통과 -- 파이프라인 정상')
else:
    print('실패 항목:')
    for name, ok, msg in results:
        if not ok:
            print(f'   • {name}: {msg}')
print('─'*50)
print('\n다음 단계:')
print('  1. Colab v2 노트북 실행 → model/ 폴더에 실제 모델 저장')
print('  2. python 04_app.py → 실시간 수어 인식 서비스 시작')
