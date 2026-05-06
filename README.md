# Whisper Dashboard

A small local HTML dashboard for transcribing audio with
[OpenAI Whisper](https://github.com/openai/whisper). It runs on your own
machine, accepts uploads such as `.m4a`, and saves transcripts as plain text.

No cloud transcription API or API key is required.

## Features

- Upload one or more audio files from the browser
- Supports `.m4a`, `.mp3`, `.mp4`, `.mpeg`, `.mpga`, `.wav`, and `.webm`
- Choose Whisper model, language, task, chunk size, timestamps, and prompt
- Watch queued/running/done/error job status in the dashboard
- Preview, copy, and download completed transcripts
- Play uploaded audio next to the transcript
- Use the same transcription engine from the command line
- Uses only Python standard-library web serving plus `openai-whisper`

## Privacy Notes

The app is designed for local use at `127.0.0.1`.

Uploaded files are stored in `uploads/`, transcripts in `transcripts/`, and job
state in `dashboard_data/`. These folders are ignored by Git, along with audio
files, generated transcript files, and `.env` files.

## Requirements

- Python 3.12 or newer
- [FFmpeg](https://ffmpeg.org/)
- Enough disk space for Whisper model files

Install FFmpeg:

```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt update && sudo apt install ffmpeg

# Windows, with Chocolatey
choco install ffmpeg
```

## Quick Start

Clone the repository and enter the folder:

```bash
git clone https://github.com/Moses-Yu/whisper-dashboard.git
cd whisper-dashboard
```

Install with Poetry:

```bash
poetry install
poetry run python dashboard.py
```

Or install with `venv` and `pip`:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python dashboard.py
```

Open the dashboard:

```text
http://127.0.0.1:7860
```

## Using The Dashboard

1. Choose or drag in one or more audio files.
2. Pick a model. `large-v3` is accurate but large; `base` is much faster for
   testing.
3. Set a language such as `Korean`, `English`, or `Auto`.
4. Click **Start transcription**.
5. Select a completed job to preview, copy, or download the transcript.

## Command Line Use

Transcribe one file directly:

```bash
poetry run python main.py "/path/to/audio.m4a" --language ko --model large-v3
```

With `venv`/`pip`:

```bash
python main.py "/path/to/audio.m4a" --language ko --model large-v3
```

Useful options:

```bash
python main.py "/path/to/audio.m4a" \
  --model large-v3 \
  --language ko \
  --chunk-seconds 300 \
  --timestamps \
  --initial-prompt "Hospital names, speaker names, and domain terms"
```

## Model Tips

Whisper downloads model files the first time you use them.

- `base`: good for quick tests
- `large-v3-turbo` or `turbo`: faster large-model option
- `large-v3`: slower, often more accurate

The default dashboard model is `large-v3`.

## Project Layout

```text
dashboard.py              Local HTML/API server
transcriber.py            Reusable Whisper transcription engine
main.py                   Command-line transcription entrypoint
templates/dashboard.html  Dashboard markup
static/dashboard.css      Dashboard styles
static/dashboard.js       Dashboard browser logic
```

## Troubleshooting

If Whisper cannot load audio, FFmpeg is usually missing or not on your `PATH`.

If the first transcription is slow, Whisper is probably downloading the selected
model. Later runs reuse the cached model.

If port `7860` is busy, run:

```bash
python dashboard.py --port 7861
```

## License

MIT
