from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import re
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
DEFAULT_DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"
SPEAKER_MODES = {"none", "segment", "sentence", "voice"}
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？])\s+")
TOKEN_ENV_NAMES = ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGINGFACEHUB_API_TOKEN")

ProgressCallback = Callable[[dict[str, Any]], None]

_MODEL_CACHE: dict[str, Any] = {}
_MODEL_LOCK = threading.Lock()
_DIARIZATION_PIPELINES: dict[tuple[str, str], Any] = {}
_DIARIZATION_LOCK = threading.Lock()


@dataclass(frozen=True)
class TranscriptionSettings:
    model: str = DEFAULT_MODEL
    language: str | None = None
    task: str = "transcribe"
    chunk_seconds: int = DEFAULT_CHUNK_SECONDS
    include_timestamps: bool = False
    speaker_mode: str = "none"
    speaker_count: int | None = None
    diarization_model: str = DEFAULT_DIARIZATION_MODEL
    initial_prompt: str | None = None


@dataclass(frozen=True)
class TranscriptSegment:
    index: int | str
    start_seconds: float
    end_seconds: float
    text: str


@dataclass(frozen=True)
class DiarizationTurn:
    start_seconds: float
    end_seconds: float
    speaker: str


@dataclass(frozen=True)
class TranscriptChunk:
    index: int
    start_seconds: float
    end_seconds: float
    text: str
    segments: tuple[TranscriptSegment, ...] = ()


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
    chunks: list[TranscriptChunk],
    include_timestamps: bool = False,
    speaker_mode: str = "none",
    diarization_turns: list[DiarizationTurn] | None = None,
) -> str:
    if speaker_mode not in SPEAKER_MODES:
        raise ValueError(f"Unknown speaker mode: {speaker_mode}")

    if speaker_mode == "voice":
        return format_voice_transcript(
            chunks,
            diarization_turns or [],
            include_timestamps,
        )

    if speaker_mode in {"segment", "sentence"}:
        return format_alternating_speaker_transcript(
            chunks,
            speaker_mode,
            include_timestamps,
        )

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


def format_alternating_speaker_transcript(
    chunks: list[TranscriptChunk],
    speaker_mode: str,
    include_timestamps: bool,
) -> str:
    turns = dialogue_turns(chunks, speaker_mode)
    labels = ("A", "B")
    lines = []
    for index, turn in enumerate(turns):
        speaker = labels[index % len(labels)]
        if include_timestamps:
            start = format_timestamp(turn.start_seconds)
            end = format_timestamp(turn.end_seconds)
            lines.append(f"[{start} - {end}] {speaker}: {turn.text}")
        else:
            lines.append(f"{speaker}: {turn.text}")
    return "\n".join(lines).strip()


def format_voice_transcript(
    chunks: list[TranscriptChunk],
    diarization_turns: list[DiarizationTurn],
    include_timestamps: bool,
) -> str:
    segments = split_segments_into_sentences(source_segments(chunks))
    if not segments:
        return ""

    speaker_labels: dict[str, str] = {}
    lines: list[TranscriptSegment] = []
    speakers: list[str] = []

    for segment in segments:
        speaker = best_speaker_for_segment(segment, diarization_turns)
        if speaker is None:
            speaker = speakers[-1] if speakers else "unknown"
        if speaker not in speaker_labels:
            speaker_labels[speaker] = speaker_label(len(speaker_labels))
        label = speaker_labels[speaker]
        speakers.append(speaker)

        if lines and lines[-1].index == label:
            previous = lines[-1]
            lines[-1] = TranscriptSegment(
                index=label,
                start_seconds=previous.start_seconds,
                end_seconds=segment.end_seconds,
                text=f"{previous.text} {segment.text}",
            )
        else:
            lines.append(
                TranscriptSegment(
                    index=label,
                    start_seconds=segment.start_seconds,
                    end_seconds=segment.end_seconds,
                    text=segment.text,
                )
            )

    blocks = []
    for turn in lines:
        if include_timestamps:
            start = format_timestamp(turn.start_seconds)
            end = format_timestamp(turn.end_seconds)
            blocks.append(f"[{start} - {end}] {turn.index}: {turn.text}")
        else:
            blocks.append(f"{turn.index}: {turn.text}")
    return "\n".join(blocks).strip()


def best_speaker_for_segment(
    segment: TranscriptSegment,
    diarization_turns: list[DiarizationTurn],
) -> str | None:
    overlaps: dict[str, float] = {}
    for turn in diarization_turns:
        overlap = min(segment.end_seconds, turn.end_seconds) - max(
            segment.start_seconds,
            turn.start_seconds,
        )
        if overlap > 0:
            overlaps[turn.speaker] = overlaps.get(turn.speaker, 0.0) + overlap

    if overlaps:
        return max(overlaps.items(), key=lambda item: item[1])[0]

    midpoint = (segment.start_seconds + segment.end_seconds) / 2
    nearest_turn = min(
        diarization_turns,
        key=lambda turn: min(
            abs(midpoint - turn.start_seconds),
            abs(midpoint - turn.end_seconds),
        ),
        default=None,
    )
    return nearest_turn.speaker if nearest_turn else None


def speaker_label(index: int) -> str:
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if index < len(alphabet):
        return alphabet[index]
    return f"S{index + 1}"


def dialogue_turns(
    chunks: list[TranscriptChunk],
    speaker_mode: str,
) -> list[TranscriptSegment]:
    if speaker_mode == "sentence":
        return split_segments_into_sentences(source_segments(chunks))
    return source_segments(chunks)


def source_segments(chunks: list[TranscriptChunk]) -> list[TranscriptSegment]:
    segments: list[TranscriptSegment] = []
    for chunk in chunks:
        if chunk.segments:
            segments.extend(chunk.segments)
        elif chunk.text:
            segments.append(
                TranscriptSegment(
                    index=len(segments) + 1,
                    start_seconds=chunk.start_seconds,
                    end_seconds=chunk.end_seconds,
                    text=chunk.text,
                )
            )
    return segments


def split_segments_into_sentences(
    segments: list[TranscriptSegment],
) -> list[TranscriptSegment]:
    turns: list[TranscriptSegment] = []
    for segment in segments:
        sentences = [
            sentence.strip()
            for sentence in SENTENCE_SPLIT_RE.split(segment.text.strip())
            if sentence.strip()
        ]
        if len(sentences) <= 1:
            turns.append(
                TranscriptSegment(
                    index=len(turns) + 1,
                    start_seconds=segment.start_seconds,
                    end_seconds=segment.end_seconds,
                    text=segment.text,
                )
            )
            continue

        duration = max(0.0, segment.end_seconds - segment.start_seconds)
        total_chars = sum(len(sentence) for sentence in sentences) or 1
        current_start = segment.start_seconds
        for sentence_index, sentence in enumerate(sentences):
            if sentence_index == len(sentences) - 1:
                current_end = segment.end_seconds
            else:
                current_end = current_start + duration * (len(sentence) / total_chars)
            turns.append(
                TranscriptSegment(
                    index=len(turns) + 1,
                    start_seconds=current_start,
                    end_seconds=current_end,
                    text=sentence,
                )
            )
            current_start = current_end
    return turns


def transcript_segments(
    result: dict[str, Any],
    chunk_offset: float,
    chunk_end: float,
    fallback_text: str,
) -> tuple[TranscriptSegment, ...]:
    raw_segments = result.get("segments") or []
    segments: list[TranscriptSegment] = []
    for raw_segment in raw_segments:
        text = str(raw_segment.get("text", "")).strip()
        if not text:
            continue
        start = chunk_offset + float(raw_segment.get("start", 0))
        end = chunk_offset + float(raw_segment.get("end", 0))
        segments.append(
            TranscriptSegment(
                index=len(segments) + 1,
                start_seconds=max(chunk_offset, start),
                end_seconds=min(chunk_end, max(start, end)),
                text=text,
            )
        )

    if not segments and fallback_text:
        segments.append(
            TranscriptSegment(
                index=1,
                start_seconds=chunk_offset,
                end_seconds=chunk_end,
                text=fallback_text,
            )
        )
    return tuple(segments)


def diarization_token(audio_path: Path) -> str:
    for name in TOKEN_ENV_NAMES:
        token = os.environ.get(name)
        if token:
            return token

    for env_path in env_file_candidates(audio_path):
        token = token_from_env_file(env_path)
        if token:
            return token

    names = " or ".join(TOKEN_ENV_NAMES)
    raise RuntimeError(
        "Voice diarization needs a Hugging Face token. Create a token, accept the "
        "pyannote model terms, then set HF_TOKEN in your environment or .env file."
        f" Checked {names}."
    )


def env_file_candidates(audio_path: Path) -> list[Path]:
    here = Path(__file__).resolve().parent
    candidates = [here / ".env", Path.cwd() / ".env", audio_path.parent / ".env"]
    unique: list[Path] = []
    for path in candidates:
        if path not in unique:
            unique.append(path)
    return unique


def token_from_env_file(env_path: Path) -> str | None:
    if not env_path.exists():
        return None

    for line in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        if name.strip() in TOKEN_ENV_NAMES:
            return value.strip().strip("\"'")
    return None


def load_diarization_pipeline(model_name: str, token: str):
    cache_key = (model_name, token)
    with _DIARIZATION_LOCK:
        pipeline = _DIARIZATION_PIPELINES.get(cache_key)
        if pipeline is not None:
            return pipeline

        try:
            from pyannote.audio import Pipeline
        except ImportError as exc:
            raise RuntimeError(
                "Voice diarization needs pyannote.audio. Install it with "
                "`pip install -r requirements-diarization.txt` or "
                "`poetry run pip install -r requirements-diarization.txt`."
            ) from exc

        try:
            pipeline = Pipeline.from_pretrained(model_name, token=token)
        except TypeError:
            pipeline = Pipeline.from_pretrained(model_name, use_auth_token=token)

        _DIARIZATION_PIPELINES[cache_key] = pipeline
        return pipeline


def diarize_audio_file(
    audio_path: Path,
    settings: TranscriptionSettings,
    progress_callback: ProgressCallback | None = None,
) -> list[DiarizationTurn]:
    emit(progress_callback, phase="loading_diarization", progress=0)
    token = diarization_token(audio_path)
    pipeline = load_diarization_pipeline(settings.diarization_model, token)
    emit(progress_callback, phase="diarizing", progress=0)

    kwargs: dict[str, Any] = {}
    if settings.speaker_count:
        kwargs["num_speakers"] = settings.speaker_count
    output = pipeline(str(audio_path), **kwargs)
    turns = normalized_diarization_turns(output)
    if not turns:
        raise RuntimeError("Voice diarization did not find any speaker turns.")

    emit(progress_callback, phase="diarization_done", progress=0)
    return turns


def normalized_diarization_turns(output: Any) -> list[DiarizationTurn]:
    diarization = getattr(output, "speaker_diarization", output)
    turns: list[DiarizationTurn] = []

    if hasattr(diarization, "itertracks"):
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            turns.append(
                DiarizationTurn(
                    start_seconds=float(turn.start),
                    end_seconds=float(turn.end),
                    speaker=str(speaker),
                )
            )
        return sorted(turns, key=lambda turn: turn.start_seconds)

    for item in diarization:
        if len(item) == 2:
            turn, speaker = item
        elif len(item) == 3:
            turn, _, speaker = item
        else:
            continue
        turns.append(
            DiarizationTurn(
                start_seconds=float(turn.start),
                end_seconds=float(turn.end),
                speaker=str(speaker),
            )
        )
    return sorted(turns, key=lambda turn: turn.start_seconds)


def transcribe_audio_file(
    audio_path: Path,
    output_path: Path | None = None,
    settings: TranscriptionSettings | None = None,
    progress_callback: ProgressCallback | None = None,
) -> TranscriptionResult:
    settings = settings or TranscriptionSettings()
    if settings.speaker_mode not in SPEAKER_MODES:
        raise ValueError(f"Unknown speaker mode: {settings.speaker_mode}")

    audio_path = Path(audio_path)
    output_path = Path(output_path) if output_path else default_output_path(audio_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if audio_path.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_AUDIO_EXTENSIONS))
        raise ValueError(f"Unsupported audio type. Supported extensions: {supported}")

    chunk_seconds = max(1, int(settings.chunk_seconds))
    diarization_turns: list[DiarizationTurn] = []
    if settings.speaker_mode == "voice":
        diarization_turns = diarize_audio_file(audio_path, settings, progress_callback)

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
        segments = transcript_segments(result, start, end, text)
        chunks.append(
            TranscriptChunk(
                index=index,
                start_seconds=start,
                end_seconds=end,
                text=text,
                segments=segments,
            )
        )
        output_path.write_text(
            format_transcript(
                chunks,
                settings.include_timestamps,
                settings.speaker_mode,
                diarization_turns,
            ),
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

    text = format_transcript(
        chunks,
        settings.include_timestamps,
        settings.speaker_mode,
        diarization_turns,
    )
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
