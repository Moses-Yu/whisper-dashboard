from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import threading
from typing import Any, Callable

import whisper
from whisper.audio import SAMPLE_RATE


SUPPORTED_AUDIO_EXTENSIONS = {
    ".m4a",
    ".mp3",
    ".mp4",
    ".mpeg",
    ".mpga",
    ".wav",
    ".webm",
}

DEFAULT_MODEL = "large-v3"
DEFAULT_CHUNK_SECONDS = 300

ProgressCallback = Callable[[dict[str, Any]], None]

_MODEL_CACHE: dict[str, Any] = {}
_MODEL_LOCK = threading.Lock()


@dataclass(frozen=True)
class TranscriptionSettings:
    model: str = DEFAULT_MODEL
    language: str | None = None
    task: str = "transcribe"
    chunk_seconds: int = DEFAULT_CHUNK_SECONDS
    include_timestamps: bool = False
    initial_prompt: str | None = None


@dataclass(frozen=True)
class TranscriptChunk:
    index: int
    start_seconds: float
    end_seconds: float
    text: str


@dataclass(frozen=True)
class TranscriptionResult:
    audio_path: Path
    output_path: Path
    duration_seconds: float
    chunks: list[TranscriptChunk]
    text: str


def available_models() -> list[str]:
    return list(whisper.available_models())


def load_cached_model(model_name: str):
    with _MODEL_LOCK:
        model = _MODEL_CACHE.get(model_name)
        if model is None:
            model = whisper.load_model(model_name)
            _MODEL_CACHE[model_name] = model
        return model


def default_output_path(audio_path: Path, output_dir: Path | None = None) -> Path:
    target_dir = output_dir or audio_path.parent
    return target_dir / f"{audio_path.stem}_result.txt"


def chunk_audio(audio, sample_rate: int, chunk_seconds: int):
    chunk_size = int(chunk_seconds * sample_rate)
    for start in range(0, len(audio), chunk_size):
        end = min(start + chunk_size, len(audio))
        yield audio[start:end], start / sample_rate, end / sample_rate


def format_timestamp(seconds: float) -> str:
    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_transcript(
    chunks: list[TranscriptChunk], include_timestamps: bool = False
) -> str:
    if not include_timestamps:
        return "\n\n".join(chunk.text for chunk in chunks if chunk.text).strip()

    blocks = []
    for chunk in chunks:
        if not chunk.text:
            continue
        start = format_timestamp(chunk.start_seconds)
        end = format_timestamp(chunk.end_seconds)
        blocks.append(f"[{start} - {end}]\n{chunk.text}")
    return "\n\n".join(blocks).strip()


def transcribe_audio_file(
    audio_path: Path,
    output_path: Path | None = None,
    settings: TranscriptionSettings | None = None,
    progress_callback: ProgressCallback | None = None,
) -> TranscriptionResult:
    settings = settings or TranscriptionSettings()
    audio_path = Path(audio_path)
    output_path = Path(output_path) if output_path else default_output_path(audio_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if audio_path.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_AUDIO_EXTENSIONS))
        raise ValueError(f"Unsupported audio type. Supported extensions: {supported}")

    chunk_seconds = max(1, int(settings.chunk_seconds))
    emit(progress_callback, phase="loading_model", progress=0)
    model = load_cached_model(settings.model)

    emit(progress_callback, phase="loading_audio", progress=0)
    audio = whisper.load_audio(str(audio_path))
    if len(audio) == 0:
        raise ValueError("Audio file appears to be empty.")

    duration_seconds = len(audio) / SAMPLE_RATE
    chunk_size = chunk_seconds * SAMPLE_RATE
    total_chunks = max(1, math.ceil(len(audio) / chunk_size))
    chunks: list[TranscriptChunk] = []

    options: dict[str, Any] = {
        "task": settings.task,
        "verbose": False,
        "fp16": getattr(model, "device", None) is not None
        and getattr(model.device, "type", "") == "cuda",
    }
    if settings.language:
        options["language"] = settings.language
    if settings.initial_prompt:
        options["initial_prompt"] = settings.initial_prompt

    for index, (chunk, start, end) in enumerate(
        chunk_audio(audio, SAMPLE_RATE, chunk_seconds), start=1
    ):
        emit(
            progress_callback,
            phase="transcribing",
            progress=(index - 1) / total_chunks,
            chunk_index=index,
            total_chunks=total_chunks,
            start_seconds=start,
        )
        result = model.transcribe(chunk, **options)
        text = result.get("text", "").strip()
        chunks.append(
            TranscriptChunk(
                index=index,
                start_seconds=start,
                end_seconds=end,
                text=text,
            )
        )
        output_path.write_text(
            format_transcript(chunks, settings.include_timestamps),
            encoding="utf-8",
        )
        emit(
            progress_callback,
            phase="chunk_done",
            progress=index / total_chunks,
            chunk_index=index,
            total_chunks=total_chunks,
            start_seconds=start,
        )

    text = format_transcript(chunks, settings.include_timestamps)
    output_path.write_text(text, encoding="utf-8")
    emit(progress_callback, phase="done", progress=1, total_chunks=total_chunks)

    return TranscriptionResult(
        audio_path=audio_path,
        output_path=output_path,
        duration_seconds=duration_seconds,
        chunks=chunks,
        text=text,
    )


def emit(progress_callback: ProgressCallback | None, **event: Any) -> None:
    if progress_callback:
        progress_callback(event)
