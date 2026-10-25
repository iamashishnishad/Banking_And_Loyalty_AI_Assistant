"""Speech-to-text with a local faster-whisper model (free, no API key, runs on CPU).

Whisper sometimes labels Hindi speech as Urdu (same spoken language) and writes it in Perso-Arabic
script, which nothing downstream handles. So the language is detected first, restricted to Hindi
or English, and passed to transcription explicitly.
"""

import asyncio
import io
import logging
import threading
import time
from dataclasses import dataclass

from app.config import get_settings

logger = logging.getLogger(__name__)

# Nudges Whisper toward banking vocabulary. The prompt's script also steers the output script, so
# Hindi gets a Devanagari prompt; a Roman one makes Whisper write romanised Hindi.
_INITIAL_PROMPTS = {
    "en": "Credit card, reward points, balance, redeem, FD, EMI, KYC.",
    "hi": "क्रेडिट कार्ड, रिवॉर्ड पॉइंट्स, बैलेंस, रिडीम, एफडी, ईएमआई, केवाईसी। मेरे पॉइंट्स कितने हैं?",
}


@dataclass
class Transcript:
    text: str
    language: str  # "hi" or "en"
    language_probability: float
    duration_s: float
    took_ms: float


class Transcriber:
    def __init__(self, model_name: str, compute_type: str) -> None:
        from faster_whisper import WhisperModel

        logger.info("Loading Whisper model %s (%s)", model_name, compute_type)
        self.model = WhisperModel(model_name, device="cpu", compute_type=compute_type)
        self._lock = threading.Lock()

    def transcribe(self, audio_bytes: bytes) -> Transcript:
        from faster_whisper import decode_audio

        started = time.perf_counter()
        audio = decode_audio(io.BytesIO(audio_bytes))  # any format PyAV/ffmpeg reads -> 16 kHz mono
        with self._lock:
            _, _, all_probs = self.model.detect_language(audio)
            probs = {lang: p for lang, p in all_probs}
            # Urdu and Hindi are one spoken language; count Urdu toward Hindi.
            hindi = probs.get("hi", 0.0) + probs.get("ur", 0.0)
            english = probs.get("en", 0.0)
            language = "hi" if hindi >= english else "en"
            segments, info = self.model.transcribe(
                audio,
                language=language,
                beam_size=5,
                vad_filter=True,
                initial_prompt=_INITIAL_PROMPTS[language],
            )
            text = " ".join(s.text.strip() for s in segments).strip()
        total = hindi + english
        return Transcript(
            text=text,
            language=language,
            language_probability=round((hindi if language == "hi" else english) / total, 3) if total else 0.0,
            duration_s=round(info.duration, 2),
            took_ms=round((time.perf_counter() - started) * 1000, 1),
        )


_transcriber: Transcriber | None = None
_transcriber_lock = threading.Lock()


def get_transcriber() -> Transcriber:
    global _transcriber
    if _transcriber is None:
        with _transcriber_lock:
            if _transcriber is None:
                s = get_settings()
                _transcriber = Transcriber(s.whisper_model, s.whisper_compute_type)
    return _transcriber


async def transcribe(audio_bytes: bytes) -> Transcript:
    return await asyncio.to_thread(lambda: get_transcriber().transcribe(audio_bytes))
