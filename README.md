HearLink — Backend Services (Flask, Whisper, Vision, SQLite)

### Overview

HearLink is a cloud‑native, production‑ready EdTech backend that turns classrooms into inclusive, intelligent learning spaces. It powers:

- Real‑time multilingual speech‑to‑text (Whisper / faster‑whisper)
- AI content automation (notes, summaries, quizzes, flashcards, exercises)
- Emotion detection and analytics from video frames (OpenCV‑based)
- Classroom chat with AI tutor and persistent chat history
- User accounts, class records, and emotion insights stored in SQLite

Recognition: Top 10 Project — Pragati META AI Hackathon 2025

Live resources (placeholders):
- [Live Web App URL](https://hearlink.vercel.app/)
- [Backend Repository](https://github.com/deep-priyo/Hearlink-Backend-Meta)
- [Frontend Repository](https://github.com/apu52/HEARLINK_META_FINAL.git) 
- [Prototype Demo Video URL](https://www.youtube.com/watch?v=St8B_a1C51A&feature=youtu.be)

There are two Flask servers provided:

- `Server.py` — main API server (default port `5009`). Includes transcription, content generation, chat endpoints, user management, and emotion insights.
- `VoiceChatbotServer.py` — auxiliary voice chatbot API (default port `5000`). Uses a custom OpenAI‑compatible LLaMA endpoint for responses and manages classroom chat history in SQLite.


### Tech Stack and Tooling

- Language: Python 3.12 (Docker base) / Python 3.x locally
- Frameworks: Flask, Flask‑CORS, Flask‑Login, Flask‑SQLAlchemy
- AI/ML: PyTorch (`torch`), Whisper / faster‑whisper, OpenCV, deepface, transformers
- LLM providers: Google Gemini (`google-generativeai`), custom OpenAI‑compatible LLaMA endpoints (via `openai`/`OpenAI` clients)
- Data: SQLite
  - Main API: `sqlite:///made_with_hardwork.db` (SQLAlchemy)
  - Chat helpers: `classroom_data.db`
- Media: FFmpeg, moviepy, pydub, PortAudio
- Package manager: pip via `requirements.txt`
- Containerization: `Dockerfile` (Ubuntu‑based Python 3.12 image)


### Requirements

System prerequisites (for local, non‑Docker runs):

- Python 3.10+ (3.12 recommended)
- FFmpeg (required by media and Whisper pipelines)
- OpenCV runtime libs (if using local camera/video processing)
- PortAudio (for certain audio features)
- Optional GPU: CUDA‑capable GPU + compatible PyTorch for faster inference (`torch.cuda.is_available()` is used to auto‑select device)

Python dependencies:

- Install from `requirements.txt`:
  - `pip install -r requirements.txt`

Data and working directories (created at runtime if missing):

- `uploads/` — uploaded media
- `results/` — generated assets (PDFs, notes, etc.)
- `registered_faces/` — face images and related assets
- `instance/` — Flask instance folder

SQLite databases used:

- Main API (`Server.py`): `sqlite:///made_with_hardwork.db`
- Chatbot helpers: `classroom_data.db`


### Environment Variables

Some features require API keys for LLMs:

- `GOOGLE_API_KEY` — for Gemini models used in translation/content generation
- `E2E_API_KEY` — for custom OpenAI‑compatible LLaMA endpoints (used in `Voice_Chat_helper.py`, `VoiceChatbotServer.py`, and parts of `ContentGeneration.py`)

How to set (Windows PowerShell):

```
$env:GOOGLE_API_KEY = "<your_key>"
$env:E2E_API_KEY = "<your_key>"
```

Or create a `.env` file in the project root (dotenv is loaded in several modules):

```
GOOGLE_API_KEY=<your_key>
E2E_API_KEY=<your_key>
```


### Setup (Local)

1) Create and activate a virtual environment

```
python -m venv .venv
\.venv\Scripts\activate  # PowerShell on Windows
```

2) Install dependencies

```
pip install --upgrade pip
pip install -r requirements.txt
```

3) Set env vars (see above), ensure FFmpeg is installed and accessible in PATH

4) Initialize runtime folders (optional — created on demand): `uploads/`, `results/`, `registered_faces/`


### Running (Local)

Run the main server (port 5009):

```
python Server.py
```

Health check:

```
curl http://localhost:5009/api/health
```

Run the voice chatbot server (port 5000):

```
python VoiceChatbotServer.py
```

Note: You can run both services in parallel on different ports if needed.


### Docker

Build the image from the provided Dockerfile:

```
docker build -t hearlink-backend:local .
```

Run the container (exposes 5009):

```
docker run --rm -p 5009:5009 ^
  -e GOOGLE_API_KEY=$env:GOOGLE_API_KEY ^
  -e E2E_API_KEY=$env:E2E_API_KEY ^
  hearlink-backend:local
```

Notes:

- The Dockerfile copies and runs `Server.py` and exposes port `5009` by default.
- System packages installed in the image include FFmpeg, OpenCV libs, PortAudio, and build tools.
- The `apt-get install` section currently lacks continuation characters in comments; ensure it builds in your Docker version. If it fails, remove inline comments or merge packages into a single line. See Troubleshooting.
- GPU support would require a CUDA‑enabled base image and matching PyTorch wheels (not covered here).


### Pipeline / Deployment

`pipeline.yaml` references a pre‑built image and sets environment variables:

```
services:
  hearlink:
    image: deeppriyo/hearlinkapp:latest
    command: ["python", "Server.py"]
    ports:
      - "5009:5009"
    environment:
      - GOOGLE_API_KEY=${GOOGLE_API_KEY}
      - E2E_API_KEY=${E2E_API_KEY}
```

Notes:

- Treat this as a service descriptor; validate syntax for your orchestrator (Docker Compose, Nomad, Kubernetes via Kompose, etc.). It may need formatting fixes.
- Ensure the image tag matches your build pipeline if you are not pulling `deeppriyo/hearlinkapp:latest`.


### Core Features

- Real‑Time Multilingual Speech‑to‑Text: Live captions across 50+ languages using Whisper
- Emotion Detection (Live & Group): Camera‑based detection of confusion, frustration, boredom; aggregated analytics
- Class Recording & Smart Notes: Auto‑generated notes and downloadable quizzes
- AI Chatbot for Learning Support: Students ask context‑aware questions in class chat
- AI‑Generated Study Materials: Notes, summaries, quizzes, flashcards, exercises in Indian languages
- Multi‑Source Content Generation: Uploads and YouTube links merged into comprehensive resources (single downloadable PDF)
- Teacher Decision Support: Real‑time engagement analytics and suggested interventions
- Device‑Agnostic Access: Web optimized; mobile app in development


### Future Roadmap

- Advanced Full‑Class Analytics: Trend dashboards for class‑wide sentiment and participation
- Mobile App: Cross‑platform Flutter app
- Smart Assistive Hardware: Affordable wearables and classroom devices
- EdTech Integrations: Google Classroom, Moodle, Microsoft Teams APIs


### Project Structure

High‑level files in the repo root:

- `Server.py` — main Flask API (port 5009). Endpoints include:
  - `GET /api/health`
  - Auth: `POST /api/login`, `POST /api/logout`, `POST /api/register`
  - Transcription: `POST /api/transcribe`, `POST /api/transcribelink`
  - Content: `GET /api/summary`, `GET /api/flashcards`, `GET /api/quiz`, `GET /api/exercise`, `GET /download/<id>/<file_type>`
  - Notes: `POST /api/generate-note`
  - Emotion/video: `GET /api/emotion_dashboard`, `POST /api/upload_video`, `GET /api/analysis/<user_id>`, `POST /api/regenerate_insights/<user_id>`, `POST /api/batch_regenerate_insights`
  - Users: `GET /api/users`, `GET /api/current_user`
  - Chat/classes: `POST /api/process-audio`, `POST /api/chat`, `GET /api/classes`, `POST /api/search-classes`, `GET /api/get-chat-history/<session_id>`, `GET /api/stats`, `GET /api/download/pdf/<filename>`

- `VoiceChatbotServer.py` — auxiliary Flask app (port 5000) for chat using a custom LLaMA endpoint
- `Voice_Chat_helper.py` — helpers for chat, DB init, audio handling
- `ContentGeneration.py` — ingest audio/video/text; Whisper/Gemini/LLaMA integrations
- `emotion_helper.py` — emotion analysis and media utilities
- `requirements.txt` — Python dependencies
- `Dockerfile` — container build for `Server.py`
- `pipeline.yaml` — service/image descriptor (see above)
- Runtime directories: `uploads/`, `results/`, `registered_faces/`
- Databases: `classroom_data.db`, `instance/made_with_hardwork.db`, `instance/sever_data.db`
- Working artifacts: `output_files/`, `summary.txt`, `flashcards.txt`, `transcript.txt`, `translated_transcript.txt`, `emotion.txt`, `detailed_notes.txt`


### Scripts and Entry Points

- Local run:
  - `python Server.py` (main API, port 5009)
  - `python VoiceChatbotServer.py` (chatbot API, port 5000)

- Docker run:
  - Entrypoint runs `Server.py` (`CMD ["python", "Server.py"]`)

No additional standalone CLI scripts; functionality is exposed via HTTP endpoints.


### Usage Examples (basic)

Transcribe an uploaded audio file:

```
curl -X POST http://localhost:5009/api/transcribe ^
  -F video=@sample.mp4 ^
  -F target_language=en
```

Transcribe from YouTube and generate notes:

```
curl -X POST http://localhost:5009/api/transcribelink ^
  -F youtube_link=https://www.youtube.com/watch?v=dQw4w9WgXcQ ^
  -F target_language=en
```

Get classes listing:

```
curl http://localhost:5009/api/classes
```

Chat message:

```
curl -X POST http://localhost:5009/api/chat ^
  -H "Content-Type: application/json" ^
  -d '{"session_id":"abc123","message":"Explain photosynthesis"}'
```

Note: Request/response shapes may evolve; inspect the route implementations for full payload details.


### Tests

No automated tests are currently included in the repository.

Suggested next steps:

- Add smoke tests for `GET /api/health` and core endpoint contracts
- Unit tests for `ContentGeneration.py` and `emotion_helper.py` (mock external APIs and model calls)
- Include sample media and golden outputs for reproducibility


### Maintainer & Role Highlights

Primary contributor: Backend development, DevOps, and frontend integration.

- Backend architecture: Designed Flask services (`Server.py`, `VoiceChatbotServer.py`), modular helpers, and data flows for STT, content generation, and emotion analytics
- Database design: Modeled users and emotion analyses in SQLite via SQLAlchemy; separate chat history DB for isolation
- AI/ML integration: Wired Whisper/faster‑whisper pipelines, Gemini for generation/translation, and custom LLaMA endpoint for chat
- DevOps: Authored Dockerfile (system deps: FFmpeg, OpenCV, PortAudio), environment management via `.env`, and deployment `pipeline.yaml`; built and published images used for live demos
- Frontend integration: Defined REST endpoints and payloads enabling teacher dashboards, class PDFs, and student chat experiences


### License

TODO: Add a `LICENSE` file (e.g., MIT, Apache‑2.0) and reference it here.


### Troubleshooting

- Docker build fails at system package install: ensure `apt-get install` lines are valid for your Docker version. Consider merging packages into one line or removing inline comments to avoid “Missing continuation character”.
- Whisper model downloads can be large; ensure network access and sufficient disk space.
- If GPU is expected but not used, verify NVIDIA drivers, CUDA toolkit, and install a CUDA build of PyTorch compatible with your base image.
