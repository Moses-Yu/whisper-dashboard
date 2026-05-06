from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from email import policy
from email.parser import BytesParser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import posixpath
import threading
import traceback
from typing import Any
from urllib.parse import quote, unquote, urlparse
import uuid

from transcriber import (
    DEFAULT_CHUNK_SECONDS,
    DEFAULT_MODEL,
    SUPPORTED_AUDIO_EXTENSIONS,
    SPEAKER_MODES,
    TranscriptionSettings,
    available_models,
    transcribe_audio_file,
)


BASE_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = BASE_DIR / "uploads"
TRANSCRIPT_DIR = BASE_DIR / "transcripts"
DASHBOARD_DATA_DIR = BASE_DIR / "dashboard_data"
JOB_STORE_PATH = DASHBOARD_DATA_DIR / "jobs.json"
MAX_UPLOAD_BYTES = 1024 * 1024 * 1024

STORE: "JobStore"
EXECUTOR: ThreadPoolExecutor


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_filename(filename: str) -> str:
    source = Path(filename).name.strip() or "audio"
    suffix = Path(source).suffix.lower()
    stem = Path(source).stem
    safe_stem = "".join(
        char if char.isalnum() or char in {" ", ".", "_", "-", "(", ")"} else "_"
        for char in stem
    ).strip(" ._")
    if not safe_stem:
        safe_stem = "audio"
    return f"{safe_stem[:90]}{suffix}"


def public_job(job: dict[str, Any], include_transcript: bool = False) -> dict[str, Any]:
    hidden = {"upload_path", "transcript_path", "traceback_path"}
    payload = {key: value for key, value in job.items() if key not in hidden}
    job_id = job["id"]
    payload["audio_url"] = f"/api/jobs/{job_id}/audio"

    transcript_path = Path(job["transcript_path"])
    if transcript_path.exists():
        payload["download_url"] = f"/api/jobs/{job_id}/download"
        text = transcript_path.read_text(encoding="utf-8", errors="replace")
        payload["preview"] = text[:600]
        if include_transcript:
            payload["transcript"] = text
    else:
        payload["preview"] = ""
        if include_transcript:
            payload["transcript"] = ""
    return payload


class JobStore:
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.RLock()
        self.jobs: dict[str, dict[str, Any]] = {}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._load()
        self._mark_interrupted_jobs()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return

        jobs = data.get("jobs", data if isinstance(data, list) else [])
        for job in jobs:
            if isinstance(job, dict) and job.get("id"):
                self.jobs[job["id"]] = job

    def _write_locked(self) -> None:
        tmp_path = self.path.with_suffix(".tmp")
        tmp_path.write_text(
            json.dumps({"jobs": list(self.jobs.values())}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp_path.replace(self.path)

    def _mark_interrupted_jobs(self) -> None:
        changed = False
        for job in self.jobs.values():
            if job.get("status") in {"queued", "running"}:
                job["status"] = "error"
                job["error"] = "Server stopped before this transcription finished."
                job["updated_at"] = utc_now()
                changed = True
        if changed:
            with self.lock:
                self._write_locked()

    def create(self, job: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            self.jobs[job["id"]] = job
            self._write_locked()
            return dict(job)

    def update(self, job_id: str, **fields: Any) -> dict[str, Any] | None:
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                return None
            job.update(fields)
            job["updated_at"] = utc_now()
            self._write_locked()
            return dict(job)

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self.lock:
            job = self.jobs.get(job_id)
            return dict(job) if job else None

    def list(self) -> list[dict[str, Any]]:
        with self.lock:
            return sorted(
                (dict(job) for job in self.jobs.values()),
                key=lambda job: job.get("created_at", ""),
                reverse=True,
            )


class DashboardServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "WhisperDashboard/1.0"

    def do_GET(self) -> None:
        try:
            parsed = urlparse(self.path)
            path = parsed.path
            if path == "/":
                self._send_file(TEMPLATE_DIR / "dashboard.html", "text/html; charset=utf-8")
            elif path.startswith("/static/"):
                self._handle_static(path)
            elif path == "/api/config":
                self._send_json(
                    {
                        "models": available_models(),
                        "default_model": DEFAULT_MODEL,
                        "default_chunk_seconds": DEFAULT_CHUNK_SECONDS,
                        "speaker_modes": sorted(SPEAKER_MODES),
                        "supported_extensions": sorted(SUPPORTED_AUDIO_EXTENSIONS),
                    }
                )
            elif path == "/api/jobs":
                self._send_json([public_job(job) for job in STORE.list()])
            elif path.startswith("/api/jobs/"):
                self._handle_job_get(path)
            else:
                self._send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self._send_json(
                {"error": "Request failed", "detail": str(exc)},
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def do_POST(self) -> None:
        try:
            parsed = urlparse(self.path)
            if parsed.path == "/api/jobs":
                self._handle_create_jobs()
            else:
                self._send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
        except ValueError as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            self._send_json(
                {"error": "Request failed", "detail": str(exc)},
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def log_message(self, format: str, *args: Any) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {self.address_string()} {format % args}")

    def _handle_static(self, path: str) -> None:
        relative = posixpath.normpath(unquote(path.removeprefix("/static/"))).lstrip("/")
        if relative.startswith("../") or relative == "..":
            self._send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
            return
        file_path = (STATIC_DIR / relative).resolve()
        if STATIC_DIR.resolve() not in file_path.parents and file_path != STATIC_DIR.resolve():
            self._send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
            return
        content_type, _ = mimetypes.guess_type(file_path)
        self._send_file(file_path, content_type or "application/octet-stream")

    def _handle_job_get(self, path: str) -> None:
        parts = [part for part in path.split("/") if part]
        if len(parts) < 3:
            self._send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
            return

        job_id = parts[2]
        action = parts[3] if len(parts) > 3 else None
        job = STORE.get(job_id)
        if not job:
            self._send_json({"error": "Job not found"}, HTTPStatus.NOT_FOUND)
            return

        if action is None:
            self._send_json(public_job(job, include_transcript=True))
        elif action == "download":
            self._send_download(job)
        elif action == "audio":
            self._send_audio(job)
        else:
            self._send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)

    def _handle_create_jobs(self) -> None:
        content_length = int(self.headers.get("Content-Length", "0") or 0)
        if content_length <= 0:
            raise ValueError("Upload body is empty.")
        if content_length > MAX_UPLOAD_BYTES:
            self._send_json(
                {"error": "Upload is too large. Maximum size is 1 GB."},
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
            )
            return

        fields, files = self._parse_multipart(content_length)
        if not files:
            raise ValueError("Choose at least one audio file.")

        settings = parse_settings(fields)
        created: list[dict[str, Any]] = []
        rejected: list[dict[str, str]] = []

        for file_item in files:
            original_filename = file_item["filename"]
            extension = Path(original_filename).suffix.lower()
            if extension not in SUPPORTED_AUDIO_EXTENSIONS:
                rejected.append(
                    {
                        "filename": original_filename,
                        "error": f"Unsupported extension: {extension or '(none)'}",
                    }
                )
                continue

            data = file_item["data"]
            if not data:
                rejected.append({"filename": original_filename, "error": "File is empty."})
                continue

            job_id = uuid.uuid4().hex[:12]
            stored_name = f"{job_id}_{safe_filename(original_filename)}"
            upload_path = UPLOAD_DIR / stored_name
            transcript_path = TRANSCRIPT_DIR / f"{Path(stored_name).stem}_result.txt"
            upload_path.write_bytes(data)

            job = {
                "id": job_id,
                "status": "queued",
                "progress": 0,
                "message": "Queued",
                "original_filename": original_filename,
                "stored_filename": stored_name,
                "size_bytes": len(data),
                "content_type": file_item["content_type"],
                "upload_path": str(upload_path),
                "transcript_path": str(transcript_path),
                "model": settings.model,
                "language": settings.language or "auto",
                "task": settings.task,
                "chunk_seconds": settings.chunk_seconds,
                "include_timestamps": settings.include_timestamps,
                "speaker_mode": settings.speaker_mode,
                "initial_prompt": settings.initial_prompt or "",
                "current_chunk": 0,
                "total_chunks": 0,
                "created_at": utc_now(),
                "updated_at": utc_now(),
            }
            STORE.create(job)
            EXECUTOR.submit(run_transcription_job, job_id)
            created.append(public_job(job))

        status = HTTPStatus.CREATED if created else HTTPStatus.BAD_REQUEST
        self._send_json({"jobs": created, "rejected": rejected}, status)

    def _parse_multipart(
        self, content_length: int
    ) -> tuple[dict[str, str], list[dict[str, Any]]]:
        content_type = self.headers.get("Content-Type", "")
        if not content_type.startswith("multipart/form-data"):
            raise ValueError("Upload must use multipart/form-data.")

        body = self.rfile.read(content_length)
        message = BytesParser(policy=policy.default).parsebytes(
            f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode()
            + body
        )
        if not message.is_multipart():
            raise ValueError("Upload body is not multipart data.")

        fields: dict[str, str] = {}
        files: list[dict[str, Any]] = []
        for part in message.iter_parts():
            if part.get_content_disposition() != "form-data":
                continue
            name = part.get_param("name", header="content-disposition")
            filename = part.get_filename()
            payload = part.get_payload(decode=True) or b""
            if filename:
                files.append(
                    {
                        "field": name,
                        "filename": filename,
                        "content_type": part.get_content_type(),
                        "data": payload,
                    }
                )
            elif name:
                charset = part.get_content_charset() or "utf-8"
                fields[name] = payload.decode(charset, errors="replace")
        return fields, files

    def _send_file(self, file_path: Path, content_type: str) -> None:
        if not file_path.exists() or not file_path.is_file():
            self._send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
            return
        data = file_path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_download(self, job: dict[str, Any]) -> None:
        transcript_path = Path(job["transcript_path"])
        if not transcript_path.exists():
            self._send_json({"error": "Transcript is not ready."}, HTTPStatus.NOT_FOUND)
            return
        data = transcript_path.read_bytes()
        filename = f"{Path(job['original_filename']).stem}_result.txt"
        encoded = quote(filename)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header(
            "Content-Disposition",
            f"attachment; filename=\"transcript.txt\"; filename*=UTF-8''{encoded}",
        )
        self.end_headers()
        self.wfile.write(data)

    def _send_audio(self, job: dict[str, Any]) -> None:
        upload_path = Path(job["upload_path"])
        if not upload_path.exists():
            self._send_json({"error": "Audio file not found."}, HTTPStatus.NOT_FOUND)
            return
        content_type, _ = mimetypes.guess_type(upload_path)
        self._send_file(upload_path, content_type or "application/octet-stream")

    def _send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def parse_settings(fields: dict[str, str]) -> TranscriptionSettings:
    model = (fields.get("model") or DEFAULT_MODEL).strip()
    if model not in available_models():
        raise ValueError(f"Unknown Whisper model: {model}")

    language = (fields.get("language") or "").strip()
    if not language or language == "auto":
        language = None

    task = (fields.get("task") or "transcribe").strip()
    if task not in {"transcribe", "translate"}:
        raise ValueError("Task must be transcribe or translate.")

    try:
        chunk_seconds = int(fields.get("chunk_seconds") or DEFAULT_CHUNK_SECONDS)
    except ValueError as exc:
        raise ValueError("Chunk seconds must be a number.") from exc
    chunk_seconds = min(max(chunk_seconds, 30), 1800)

    include_timestamps = (fields.get("include_timestamps") or "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    speaker_mode = (fields.get("speaker_mode") or "none").strip()
    if speaker_mode not in SPEAKER_MODES:
        raise ValueError("Speaker mode must be none, segment, or sentence.")

    initial_prompt = (fields.get("initial_prompt") or "").strip() or None

    return TranscriptionSettings(
        model=model,
        language=language,
        task=task,
        chunk_seconds=chunk_seconds,
        include_timestamps=include_timestamps,
        speaker_mode=speaker_mode,
        initial_prompt=initial_prompt,
    )


def run_transcription_job(job_id: str) -> None:
    job = STORE.get(job_id)
    if not job:
        return

    settings = TranscriptionSettings(
        model=job["model"],
        language=None if job["language"] == "auto" else job["language"],
        task=job["task"],
        chunk_seconds=int(job["chunk_seconds"]),
        include_timestamps=bool(job["include_timestamps"]),
        speaker_mode=job.get("speaker_mode", "none"),
        initial_prompt=job.get("initial_prompt") or None,
    )

    STORE.update(
        job_id,
        status="running",
        started_at=utc_now(),
        progress=0,
        message="Starting",
    )

    def on_progress(event: dict[str, Any]) -> None:
        phase = event.get("phase")
        fields: dict[str, Any] = {
            "progress": round(float(event.get("progress", 0)), 4),
            "message": progress_message(event),
        }
        if "chunk_index" in event:
            fields["current_chunk"] = event["chunk_index"]
        if "total_chunks" in event:
            fields["total_chunks"] = event["total_chunks"]
        if phase == "done":
            fields["progress"] = 1
        STORE.update(job_id, **fields)

    try:
        result = transcribe_audio_file(
            Path(job["upload_path"]),
            Path(job["transcript_path"]),
            settings,
            on_progress,
        )
        STORE.update(
            job_id,
            status="done",
            progress=1,
            message="Complete",
            finished_at=utc_now(),
            duration_seconds=round(result.duration_seconds, 2),
            text_length=len(result.text),
            chunk_count=len(result.chunks),
        )
    except Exception as exc:
        trace_path = Path(job["transcript_path"]).with_suffix(".error.txt")
        trace_path.write_text(traceback.format_exc(), encoding="utf-8")
        STORE.update(
            job_id,
            status="error",
            message="Failed",
            error=str(exc),
            traceback_path=str(trace_path),
            finished_at=utc_now(),
        )


def progress_message(event: dict[str, Any]) -> str:
    phase = event.get("phase")
    if phase == "loading_model":
        return "Loading model"
    if phase == "loading_audio":
        return "Loading audio"
    if phase == "transcribing":
        return f"Chunk {event.get('chunk_index')}/{event.get('total_chunks')}"
    if phase == "chunk_done":
        return f"Finished chunk {event.get('chunk_index')}/{event.get('total_chunks')}"
    if phase == "done":
        return "Complete"
    return "Working"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local Whisper HTML dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    for directory in (UPLOAD_DIR, TRANSCRIPT_DIR, DASHBOARD_DATA_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    global STORE, EXECUTOR
    STORE = JobStore(JOB_STORE_PATH)
    EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="whisper-job")

    server = DashboardServer((args.host, args.port), DashboardHandler)
    print(f"Whisper dashboard running at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping dashboard...")
    finally:
        server.server_close()
        EXECUTOR.shutdown(wait=False, cancel_futures=False)


if __name__ == "__main__":
    main()
