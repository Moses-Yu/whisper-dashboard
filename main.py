from __future__ import annotations

import argparse
from pathlib import Path

from transcriber import (
    DEFAULT_CHUNK_SECONDS,
    DEFAULT_MODEL,
    TranscriptionSettings,
    default_output_path,
    transcribe_audio_file,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Transcribe an audio file with Whisper.")
    parser.add_argument("audio", type=Path, help="Path to an audio file, such as .m4a.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Where to write the transcript. Defaults to <audio>_result.txt.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Whisper model name.")
    parser.add_argument(
        "--language",
        default=None,
        help="Optional language code, for example ko or en. Omit for auto-detect.",
    )
    parser.add_argument(
        "--task",
        default="transcribe",
        choices=["transcribe", "translate"],
        help="Whisper task.",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=int,
        default=DEFAULT_CHUNK_SECONDS,
        help="Chunk size in seconds.",
    )
    parser.add_argument(
        "--timestamps",
        action="store_true",
        help="Include chunk timestamps in the transcript.",
    )
    parser.add_argument(
        "--speaker-mode",
        default="none",
        choices=["none", "segment", "sentence", "voice"],
        help="Optional A/B speaker labels using local post-processing.",
    )
    parser.add_argument(
        "--speaker-count",
        type=int,
        default=None,
        help="Expected number of voices for --speaker-mode voice. Use 2 for calls.",
    )
    parser.add_argument(
        "--initial-prompt",
        default=None,
        help="Optional Whisper prompt to guide names, terms, or context.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = args.output or default_output_path(args.audio)

    settings = TranscriptionSettings(
        model=args.model,
        language=args.language,
        task=args.task,
        chunk_seconds=args.chunk_seconds,
        include_timestamps=args.timestamps,
        speaker_mode=args.speaker_mode,
        speaker_count=args.speaker_count,
        initial_prompt=args.initial_prompt,
    )

    def show_progress(event):
        phase = event.get("phase")
        if phase == "loading_model":
            print(f"Loading Whisper model: {settings.model}")
        elif phase == "loading_audio":
            print(f"Loading audio: {args.audio}")
        elif phase == "loading_diarization":
            print("Loading speaker diarization model...")
        elif phase == "diarizing":
            print("Detecting speakers by voice...")
        elif phase == "transcribing":
            chunk = event.get("chunk_index")
            total = event.get("total_chunks")
            start = event.get("start_seconds", 0)
            print(f"Transcribing chunk {chunk}/{total} starting at {start:.1f}s...")

    result = transcribe_audio_file(args.audio, output, settings, show_progress)
    print(f"Transcript written to {result.output_path}")


if __name__ == "__main__":
    main()
