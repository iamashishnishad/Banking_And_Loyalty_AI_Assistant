"""Voice input: speech-to-text with local Whisper, optionally followed by a chat turn."""

import logging
from dataclasses import asdict

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from app.agent.llm import get_llm
from app.api.chat import ensure_customer, run_chat
from app.api.deps import RedisDep, SessionDep, TenantDep
from app.config import get_settings
from app.schemas import TranscriptOut, VoiceChatOut
from app.services.speech import Transcript, transcribe

router = APIRouter(prefix="/voice", tags=["voice"])
logger = logging.getLogger(__name__)


async def _transcribe_upload(audio: UploadFile) -> Transcript:
    settings = get_settings()
    if not settings.voice_enabled:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Voice input is disabled")
    max_bytes = settings.max_audio_mb * 1024 * 1024
    data = await audio.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, f"Audio exceeds {settings.max_audio_mb} MB")
    if not data:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Empty audio file")
    try:
        transcript = await transcribe(data)
    except Exception as exc:  # PyAV raises a family of format errors; all mean "can't decode this"
        logger.warning("Audio decode/transcription failed", exc_info=True)
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "Could not decode audio") from exc
    if not transcript.text:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "No speech detected")
    return transcript


@router.post("/transcribe", response_model=TranscriptOut)
async def transcribe_audio(audio: UploadFile = File(..., description="wav, mp3, m4a, ogg, webm ...")) -> TranscriptOut:
    return TranscriptOut(**asdict(await _transcribe_upload(audio)))


@router.post("/chat", response_model=VoiceChatOut)
async def voice_chat(
    session: SessionDep,
    tenant_id: TenantDep,
    redis: RedisDep,
    customer_ref: str = Form(...),
    audio: UploadFile = File(...),
) -> VoiceChatOut:
    await ensure_customer(session, tenant_id, customer_ref)
    get_llm()  # fail fast on a missing key before spending seconds on transcription
    transcript = await _transcribe_upload(audio)
    reply = await run_chat(redis, tenant_id, customer_ref, transcript.text)
    return VoiceChatOut(transcript=TranscriptOut(**asdict(transcript)), reply=reply)
