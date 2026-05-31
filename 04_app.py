import sys
import time
import datetime
import cv2
import numpy as np
import platform
from pathlib import Path
from io import BytesIO

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QLabel, QVBoxLayout, QHBoxLayout,
    QPushButton, QFrame, QSizePolicy,
    QTextEdit, QFileDialog, QMessageBox,
)
from PySide6.QtCore import Qt, QTimer, QThread, Signal
from PySide6.QtGui import QImage, QPixmap, QFont

# ── TTS: pyttsx3(오프라인) → gTTS+pygame 순서로 시도 ─────────
try:
    import pyttsx3 as _pyttsx3
    TTS_BACKEND   = 'pyttsx3'
    TTS_AVAILABLE = True
except Exception:
    try:
        from gtts import gTTS
        import pygame
        pygame.mixer.init()
        TTS_BACKEND   = 'gtts'
        TTS_AVAILABLE = True
    except Exception:
        TTS_BACKEND   = None
        TTS_AVAILABLE = False

# ── STT ──────────────────────────────────────────────────────
try:
    import speech_recognition as sr
    STT_AVAILABLE = True
except Exception:
    STT_AVAILABLE = False

import importlib.util
_spec = importlib.util.spec_from_file_location('inference', '03_inference.py')
_mod  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
SignInferencer = _mod.SignInferencer

# 실생활 시나리오 시뮬레이션 시퀀스 (카메라/모델 없이 UI 테스트용)
_SCENARIOS = {
    '병원 시나리오': ['안녕하세요', '아파요', '도와주세요', '감사합니다'],
    '일상 대화':     ['안녕하세요', '괜찮아요', '네', '감사합니다'],
    '식사 상황':     ['밥', '물', '좋아요', '감사합니다'],
    '긴급 상황':     ['도와주세요', '아파요', '병원'],
}


def _get_font(size, bold=False):
    font = QFont()
    font.setFamily('맑은 고딕' if platform.system() == 'Windows' else 'NanumGothic')
    font.setPointSize(size)
    font.setBold(bold)
    return font


class TTSThread(QThread):
    def __init__(self, text):
        super().__init__()
        self.text = text

    def run(self):
        if not TTS_AVAILABLE:
            return
        try:
            if TTS_BACKEND == 'pyttsx3':
                engine = _pyttsx3.init()
                engine.setProperty('rate', 160)
                engine.say(self.text)
                engine.runAndWait()
            else:
                tts = gTTS(text=self.text, lang='ko')
                buf = BytesIO()
                tts.write_to_fp(buf)
                buf.seek(0)
                pygame.mixer.music.load(buf, 'mp3')
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.05)
        except Exception:
            pass


class STTThread(QThread):
    result_ready = Signal(str)
    listening    = Signal(bool)

    def run(self):
        if not STT_AVAILABLE:
            self.result_ready.emit('[speech_recognition 패키지 없음]')
            return
        r   = sr.Recognizer()
        mic = sr.Microphone()
        self.listening.emit(True)
        try:
            with mic as source:
                r.adjust_for_ambient_noise(source, duration=0.3)
                audio = r.listen(source, timeout=8, phrase_time_limit=10)
            text = r.recognize_google(audio, language='ko-KR')
            self.result_ready.emit(text)
        except sr.WaitTimeoutError:
            self.result_ready.emit('[시간 초과]')
        except sr.UnknownValueError:
            self.result_ready.emit('[인식 실패]')
        except Exception as e:
            self.result_ready.emit(f'[오류: {e}]')
        finally:
            self.listening.emit(False)


# 시뮬레이션용 타이머 스레드
class SimThread(QThread):
    word_ready = Signal(str)

    def __init__(self, words, interval=1.5):
        super().__init__()
        self.words    = words
        self.interval = interval

    def run(self):
        for w in self.words:
            self.word_ready.emit(w)
            time.sleep(self.interval)


class SignLanguageApp(QMainWindow):
    BUFFER_SIZE    = 12
    CONFIRM_FRAMES = 8
    TTS_COOLDOWN   = 2.0

    def __init__(self, simulate: bool = False):
        super().__init__()
        self.simulate = simulate
        self.setWindowTitle('한국수어 ↔ 음성 실시간 의사소통' + (' [시뮬레이션]' if simulate else ''))
        self.setMinimumSize(1200, 720)
        self.setStyleSheet('background-color: #1a1a2e;')

        self.cap         = None
        self.inferencer  = None
        self.sim_thread  = None

        if not simulate:
            try:
                self.inferencer = SignInferencer()
            except FileNotFoundError as e:
                self._show_no_model_ui(str(e))
                return
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self._show_error('웹캠을 찾을 수 없습니다.')
                return

        self.word_buffer   = []
        self.last_word     = ''
        self.last_tts_word = ''
        self.last_tts_time = 0.0
        self.tts_thread    = None
        self.stt_thread    = None
        self.sentence      = []
        self.conv_log      = []   # 전체 대화 기록 [(시각, 화자, 내용)]

        self._build_ui()

        if not simulate:
            self.timer = QTimer()
            self.timer.timeout.connect(self._update_frame)
            self.timer.start(33)

    # ── UI ───────────────────────────────────────────────────
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main = QHBoxLayout(central)
        main.setContentsMargins(12, 12, 12, 12)
        main.setSpacing(12)

        # 왼쪽: 카메라 or 시뮬레이션
        left = QVBoxLayout()
        if self.simulate:
            sim_frame = QFrame()
            sim_frame.setStyleSheet(self._card_style())
            sl = QVBoxLayout(sim_frame)
            sl.addWidget(self._chip('시뮬레이션 모드 — 실생활 시나리오 테스트'))

            for name, words in _SCENARIOS.items():
                btn = QPushButton(f'▶  {name}  ({" → ".join(words)})')
                btn.setFont(_get_font(10))
                btn.setStyleSheet(self._btn_style('#1a3a6b', '#2471a3'))
                btn.clicked.connect(lambda _, w=words: self._run_sim(w))
                sl.addWidget(btn)

            self.lbl_sim_status = QLabel('시나리오를 선택해 수어 단어 흐름을 테스트하세요.')
            self.lbl_sim_status.setFont(_get_font(11))
            self.lbl_sim_status.setStyleSheet('color:#aaaacc; border:none;')
            self.lbl_sim_status.setWordWrap(True)
            sl.addWidget(self.lbl_sim_status)
            sl.addStretch()
            left.addWidget(sim_frame)
        else:
            self.cam_label = QLabel()
            self.cam_label.setAlignment(Qt.AlignCenter)
            self.cam_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self.cam_label.setStyleSheet(
                'background:#0f0f23; border-radius:12px; border:2px solid #2d2d5e;'
            )
            left.addWidget(self.cam_label)

        main.addLayout(left, stretch=3)

        # 오른쪽 패널
        right = QVBoxLayout()
        right.setSpacing(8)
        main.addLayout(right, stretch=2)

        right.addWidget(self._word_panel())
        right.addWidget(self._sign_sentence_panel())
        right.addWidget(self._stt_panel())
        right.addWidget(self._conv_log_panel())

        btn_row = QHBoxLayout()
        btn_export = QPushButton('대화 저장')
        btn_export.setFont(_get_font(10))
        btn_export.setStyleSheet(self._btn_style('#1a4a2a', '#27ae60'))
        btn_export.clicked.connect(self._export_conv)
        btn_row.addWidget(btn_export)

        btn_quit = QPushButton('종료')
        btn_quit.setFont(_get_font(10))
        btn_quit.setStyleSheet(self._btn_style('#3d0000', '#7d0000'))
        btn_quit.clicked.connect(self.close)
        btn_row.addWidget(btn_quit)
        right.addLayout(btn_row)

    def _word_panel(self):
        frame = QFrame()
        frame.setStyleSheet(self._card_style())
        layout = QVBoxLayout(frame)
        layout.addWidget(self._chip('현재 인식'))

        self.lbl_word = QLabel('대기 중...')
        self.lbl_word.setFont(_get_font(28, bold=True))
        self.lbl_word.setStyleSheet('color:#e0e0ff; border:none;')
        self.lbl_word.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.lbl_word)

        self.lbl_conf = QLabel('')
        self.lbl_conf.setFont(_get_font(10))
        self.lbl_conf.setStyleSheet('color:#6688aa; border:none;')
        self.lbl_conf.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.lbl_conf)

        row = QHBoxLayout()
        btn_add = QPushButton('+ 문장에 추가')
        btn_add.setFont(_get_font(10, bold=True))
        btn_add.setStyleSheet(self._btn_style('#1a6b3a', '#27ae60'))
        btn_add.clicked.connect(self._add_to_sentence)
        row.addWidget(btn_add)

        btn_tts = QPushButton('🔊 단어 읽기')
        btn_tts.setFont(_get_font(10))
        btn_tts.setStyleSheet(self._btn_style('#0f3460', '#1a5276'))
        btn_tts.clicked.connect(self._speak_current)
        row.addWidget(btn_tts)
        layout.addLayout(row)

        self.top3_labels = []
        for _ in range(3):
            lbl = QLabel('')
            lbl.setFont(_get_font(10))
            lbl.setStyleSheet('color:#8888aa; border:none;')
            lbl.setAlignment(Qt.AlignCenter)
            layout.addWidget(lbl)
            self.top3_labels.append(lbl)

        return frame

    def _sign_sentence_panel(self):
        frame = QFrame()
        frame.setStyleSheet(self._card_style())
        layout = QVBoxLayout(frame)
        layout.addWidget(self._chip('🤟 수어 → 문장'))

        self.lbl_sentence = QLabel('(단어를 인식하고 "+ 문장에 추가"를 누르세요)')
        self.lbl_sentence.setFont(_get_font(11))
        self.lbl_sentence.setStyleSheet('color:#ccccee; border:none; padding:4px;')
        self.lbl_sentence.setAlignment(Qt.AlignCenter)
        self.lbl_sentence.setWordWrap(True)
        self.lbl_sentence.setMinimumHeight(44)
        layout.addWidget(self.lbl_sentence)

        row = QHBoxLayout()
        for label, cb, bg, hv in [
            ('🔊 문장 읽기', self._speak_sentence,  '#0f3460', '#1a5276'),
            ('← 삭제',       self._remove_last_word, '#3d2200', '#7d4800'),
            ('지우기',        self._clear_sentence,   '#3d0000', '#7d0000'),
        ]:
            btn = QPushButton(label)
            btn.setFont(_get_font(10))
            btn.setStyleSheet(self._btn_style(bg, hv))
            btn.clicked.connect(cb)
            row.addWidget(btn)
        layout.addLayout(row)
        return frame

    def _stt_panel(self):
        frame = QFrame()
        frame.setStyleSheet(self._card_style())
        layout = QVBoxLayout(frame)
        layout.addWidget(self._chip('🎤 상대방 말 → 텍스트  (청인 → 청각장애인)'))

        self.txt_stt = QTextEdit()
        self.txt_stt.setReadOnly(True)
        self.txt_stt.setFont(_get_font(11))
        self.txt_stt.setStyleSheet(
            'background:#0f1a2e; color:#e8f4fd; border-radius:8px; '
            'border:1px solid #1a5276; padding:6px;'
        )
        self.txt_stt.setPlaceholderText('상대방이 말하면 텍스트로 표시됩니다.')
        self.txt_stt.setMaximumHeight(80)
        layout.addWidget(self.txt_stt)

        row = QHBoxLayout()
        self.btn_mic = QPushButton('🎤 듣기 시작')
        self.btn_mic.setFont(_get_font(10, bold=True))
        self.btn_mic.setStyleSheet(self._btn_style('#1a3a6b', '#2471a3'))
        self.btn_mic.clicked.connect(self._start_stt)
        row.addWidget(self.btn_mic)

        btn_clr = QPushButton('지우기')
        btn_clr.setFont(_get_font(10))
        btn_clr.setStyleSheet(self._btn_style('#3d0000', '#7d0000'))
        btn_clr.clicked.connect(self.txt_stt.clear)
        row.addWidget(btn_clr)
        layout.addLayout(row)
        return frame

    def _conv_log_panel(self):
        frame = QFrame()
        frame.setStyleSheet(self._card_style())
        layout = QVBoxLayout(frame)
        layout.addWidget(self._chip('대화 기록'))

        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setFont(_get_font(9))
        self.txt_log.setStyleSheet(
            'background:#0a1020; color:#aaaacc; border-radius:8px; '
            'border:1px solid #1a2a4e; padding:4px;'
        )
        self.txt_log.setMaximumHeight(70)
        layout.addWidget(self.txt_log)
        return frame

    # ── 프레임 업데이트 (실제 모드) ──────────────────────────
    def _update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        frame = cv2.flip(frame, 1)
        word, conf, top3 = self.inferencer.predict(frame)

        if word:
            self.word_buffer.append(word)
        if len(self.word_buffer) > self.BUFFER_SIZE:
            self.word_buffer.pop(0)
        if self.word_buffer:
            candidate = max(set(self.word_buffer), key=self.word_buffer.count)
            if self.word_buffer.count(candidate) >= self.CONFIRM_FRAMES:
                if candidate != self.last_word:
                    self.last_word = candidate
                    self._on_word_confirmed(candidate, conf)

        frame = self.inferencer.draw_landmarks(frame)
        for i, lbl in enumerate(self.top3_labels):
            lbl.setText(f'{i+1}. {top3[i][0]}  ({top3[i][1]*100:.0f}%)' if i < len(top3) else '')

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch  = frame_rgb.shape
        qimg = QImage(frame_rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix  = QPixmap.fromImage(qimg).scaled(
            self.cam_label.width(), self.cam_label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation,
        )
        self.cam_label.setPixmap(pix)

    # ── 시뮬레이션 모드 ──────────────────────────────────────
    def _run_sim(self, words):
        if self.sim_thread and self.sim_thread.isRunning():
            return
        self.lbl_sim_status.setText(f'실행 중: {" → ".join(words)}')
        self.sim_thread = SimThread(words, interval=1.5)
        self.sim_thread.word_ready.connect(self._on_sim_word)
        self.sim_thread.finished.connect(
            lambda: self.lbl_sim_status.setText('시뮬레이션 완료. 위 버튼으로 다시 실행하거나 다른 시나리오를 선택하세요.')
        )
        self.sim_thread.start()

    def _on_sim_word(self, word):
        self.last_word = word
        self._on_word_confirmed(word, 0.95)
        self._add_to_sentence()

    # ── 수어 관련 ─────────────────────────────────────────────
    def _on_word_confirmed(self, word, conf):
        self.lbl_word.setText(word)
        self.lbl_conf.setText(f'신뢰도 {conf*100:.1f}%')
        color = '#00e676' if conf >= 0.8 else '#ffeb3b' if conf >= 0.6 else '#ff7043'
        self.lbl_word.setStyleSheet(f'color:{color}; border:none;')
        now = time.time()
        if word != self.last_tts_word or (now - self.last_tts_time) > self.TTS_COOLDOWN:
            self.last_tts_word = word
            self.last_tts_time = now
            self._speak(word)

    def _add_to_sentence(self):
        if self.last_word:
            self.sentence.append(self.last_word)
            self._refresh_sentence()
            self.word_buffer.clear()
            self.last_word = ''

    def _remove_last_word(self):
        if self.sentence:
            self.sentence.pop()
            self._refresh_sentence()

    def _clear_sentence(self):
        self.sentence.clear()
        self._refresh_sentence()

    def _refresh_sentence(self):
        txt = '  '.join(self.sentence) if self.sentence else '(단어를 인식하고 "+ 문장에 추가"를 누르세요)'
        self.lbl_sentence.setText(txt)

    def _speak_sentence(self):
        if self.sentence:
            full = ' '.join(self.sentence)
            self._log_conv('수어(나)', full)
            self._speak(full)

    def _speak_current(self):
        if self.last_word:
            self._speak(self.last_word)

    def _speak(self, text):
        if not TTS_AVAILABLE:
            return
        if self.tts_thread and self.tts_thread.isRunning():
            return
        self.tts_thread = TTSThread(text)
        self.tts_thread.start()

    # ── STT ──────────────────────────────────────────────────
    def _start_stt(self):
        if self.stt_thread and self.stt_thread.isRunning():
            return
        self.btn_mic.setText('🎙 듣는 중...')
        self.btn_mic.setEnabled(False)
        self.stt_thread = STTThread()
        self.stt_thread.result_ready.connect(self._on_stt_result)
        self.stt_thread.listening.connect(
            lambda active: (self.btn_mic.setText('🎤 듣기 시작'), self.btn_mic.setEnabled(True))
            if not active else None
        )
        self.stt_thread.start()

    def _on_stt_result(self, text: str):
        if text and not text.startswith('['):
            self.txt_stt.append(f'🗣 {text}')
            self._log_conv('상대방(음성)', text)

    # ── 대화 기록 ─────────────────────────────────────────────
    def _log_conv(self, speaker: str, text: str):
        ts = datetime.datetime.now().strftime('%H:%M:%S')
        entry = f'[{ts}] {speaker}: {text}'
        self.conv_log.append(entry)
        self.txt_log.append(entry)

    def _export_conv(self):
        if not self.conv_log:
            QMessageBox.information(self, '알림', '저장할 대화 기록이 없습니다.')
            return
        path, _ = QFileDialog.getSaveFileName(
            self, '대화 저장', f'대화_{datetime.datetime.now():%Y%m%d_%H%M%S}.txt',
            'Text Files (*.txt)'
        )
        if path:
            with open(path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(self.conv_log))
            QMessageBox.information(self, '저장 완료', f'저장됨: {path}')

    # ── 유틸 ─────────────────────────────────────────────────
    @staticmethod
    def _card_style():
        return 'background:#16213e; border-radius:12px; border:2px solid #0f3460;'

    @staticmethod
    def _btn_style(bg, hover):
        return (f'QPushButton {{background:{bg}; color:#e0e0ff; border-radius:8px; '
                f'padding:7px; border:1px solid {hover};}}'
                f'QPushButton:hover {{background:{hover};}}')

    @staticmethod
    def _chip(text):
        lbl = QLabel(text)
        lbl.setFont(_get_font(10))
        lbl.setStyleSheet('color:#8888aa; border:none;')
        lbl.setAlignment(Qt.AlignCenter)
        return lbl

    def _show_no_model_ui(self, msg):
        lbl = QLabel(
            f'모델 파일이 없습니다.\n\n{msg}\n\n'
            'Colab 학습 후 model/ 폴더에 파일을 넣고 재실행하세요.\n\n'
            '팁: python 04_app.py --simulate 로 시뮬레이션 모드 실행 가능'
        )
        lbl.setFont(_get_font(13))
        lbl.setStyleSheet('color:#ff7043; background:#1a1a2e;')
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setWordWrap(True)
        self.setCentralWidget(lbl)

    def _show_error(self, msg):
        lbl = QLabel(msg)
        lbl.setFont(_get_font(14))
        lbl.setStyleSheet('color:#ff7043; background:#1a1a2e;')
        lbl.setAlignment(Qt.AlignCenter)
        self.setCentralWidget(lbl)

    def closeEvent(self, event):
        if hasattr(self, 'timer'):
            self.timer.stop()
        if self.cap and self.cap.isOpened():
            self.cap.release()
        if self.inferencer:
            self.inferencer.close()
        event.accept()


if __name__ == '__main__':
    simulate = '--simulate' in sys.argv
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    win = SignLanguageApp(simulate=simulate)
    win.show()
    sys.exit(app.exec())
