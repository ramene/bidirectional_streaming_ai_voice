# Local $0 Apple-Silicon variant

A fully-local, **zero-API-cost** variation of this two-way voice loop, for Apple Silicon
(M-series) Macs — plus a **browser client** so you can talk to it from an iPad / iPhone / any
browser instead of the desktop `pygame` + spacebar path.

It keeps the original design's spirit (mic → STT → LLM → **streamed** TTS → playback,
push-to-talk turns) but swaps every cloud vendor for an on-device model:

| Stage | Original (main / `assembly_api_transcription`) | This variant |
|---|---|---|
| STT | faster-whisper / AssemblyAI (cloud) | **MLX-Whisper** `whisper-large-v3-turbo` (on-device) |
| LLM | Anthropic Claude (cloud) | **any OpenAI-compatible local server** (Ollama / LM Studio / vLLM / llama.cpp); optional Anthropic arm for hard turns |
| TTS | ElevenLabs (cloud) | **Kokoro-82M** via `mlx-audio` (on-device); macOS `say` fallback |
| UI / turns | desktop `pygame` + spacebar | **browser** (iPad / iPhone / desktop), push-to-talk, WebSocket audio streaming |

Net result: no API keys required, nothing leaves the machine, `$0` per turn.

## Hands-free & fluid
- **Wake word** — flip *Always-on* and say **"Charlie"** (configurable via `WAKE_WORD`). Client-side VAD calibrates to your room's noise floor, then segments on a ~1s pause. Audible turn cues: rising beep = awake/your-turn, single beep = got-it, falling beep = asleep. Sleep phrases: "stop listening", "go to sleep", etc.
- **Barge-in** — talk over the reply (or press the button) to interrupt it instantly; the server cancels the in-flight turn.
- **Robust STT** — English-locked (Whisper was auto-detecting the wrong language and garbling), self-conditioning off (kills runaway repetition loops), plus punctuation/degenerate-output guards so room noise doesn't create junk turns.
- **British (or any) voice** — Kokoro ships US + British voices; default `bm_george` (British male). Swap with `KOKORO_VOICE`.
- **Intent-aware routing** — "ideate on X" stays generative instead of being forced into fact-recall; a `CONTEXT_PROVIDER` hook grounds turns when you want facts.
- **Transcript capture** — every turn is written to `transcripts/session-*.jsonl`.

## Why a server + browser client
The original runs on one desktop. Routing STT/LLM/TTS through a small FastAPI + WebSocket
server lets a phone or tablet be a **thin capture/playback client** while the Mac does all the
inference. The turn protocol emits discrete events (`user_text`, `reply_text`, per-sentence
`audio`, `turn_end`) so a lip-synced talking-head avatar can subscribe later.

## The memory hook (the README's #1 "future angle")
The original notes that a **better memory hierarchy** is the top wanted improvement (it currently
hand-maintains `summaries.py`). This variant exposes that as a pluggable seam: set
`CONTEXT_PROVIDER="yourmodule:yourfunc"` and your function `(user_text) -> str` is injected into
the system prompt each turn. Drop in any RAG / vector store / memory system; the loop stays generic.

## Setup
```bash
python3.11 -m venv .venv && . .venv/bin/activate        # Python >=3.10 required (mlx-audio)
pip install -r requirements.txt
brew install espeak-ng ffmpeg                            # Kokoro g2p + audio decode
# start a local LLM, e.g.:  ollama serve  &&  ollama pull llama3.1:8b
```

## Run
```bash
export LLM_BASE_URL=http://127.0.0.1:11434/v1   # Ollama (default); or LM Studio :1234/v1, vLLM, etc.
export LLM_MODEL=llama3.1:8b
# optional: export ANTHROPIC_API_KEY=...        # routes hard/complex turns to Claude (~20%)
# optional: export CONTEXT_PROVIDER=mymem:lookup # your RAG/memory hook
python server.py 8600 <your-token>
```
Open `http://127.0.0.1:8600/?token=<your-token>` on the same Mac (mic works on `localhost`).
Hold the button, speak, release.

> **Mic + remote devices:** browsers only allow microphone capture over `https://` **or**
> `localhost`. To reach it from a phone, put it behind TLS (a reverse proxy, or `tailscale serve`
> which provides HTTPS on your tailnet) and open the `https://` URL there.

## Config (env)
| Var | Default | Meaning |
|---|---|---|
| `LLM_BASE_URL` | `http://127.0.0.1:11434/v1` | OpenAI-compatible LLM base |
| `LLM_MODEL` | `llama3.1:8b` | model name on that server |
| `LLM_API_KEY` | – | bearer for the LLM server (if it needs one) |
| `ANTHROPIC_API_KEY` | – | if set, hard turns route to Claude |
| `WHISPER_REPO` | `mlx-community/whisper-large-v3-turbo` | STT model |
| `KOKORO_VOICE` | `af_heart` | Kokoro voice |
| `CONTEXT_PROVIDER` | – | `module:function` memory/RAG hook |

## Measured latency (Apple M4 Max, 36 GB)
Local models, on-device, per turn:

| Turn | STT | LLM | TTS (first audio) | Total |
|---|---|---|---|---|
| Short chit-chat (warm) | ~1.8 s | ~3.4 s | ~0.1 s | **~5.4 s** |
| Long grounded answer (large model + RAG) | ~1.4 s | ~24 s | ~5 s* | ~31 s |

*first TTS included a one-time Kokoro warmup. STT (MLX-Whisper) is far faster than the original's
~4.5 s. The LLM dominates long turns — a smaller/faster local model is the main latency lever.

## Notes
- **Memory footprint:** Whisper (~1.5 GB) + a local LLM + Kokoro (~0.3 GB) are held together; keep
  total model residency under your unified-memory budget (don't co-load two large models).
- Kokoro requires `misaki[en]` (its g2p) and `espeak-ng`. `say` is the automatic fallback if Kokoro
  can't load, so the loop still works out of the box.
- Thanks to @ccappetta for the original design — this variant only swaps the vendors for local models
  and adds a browser client + a memory seam.
