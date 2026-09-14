#!/usr/bin/env python3
"""Local $0 Apple-Silicon variant — bidirectional streaming voice, browser client.

A fully-local, zero-API-cost variation of the two-way voice loop, for Apple Silicon
(M-series) Macs. It keeps the original design's spirit (mic -> STT -> LLM -> streamed
TTS -> playback, push-to-talk turns) but swaps every cloud vendor for a local model:

    STT  : AssemblyAI / faster-whisper   ->  MLX-Whisper (mlx-whisper, on-device)
    LLM  : Anthropic (cloud)             ->  any OpenAI-compatible local server
                                            (Ollama / LM Studio / vLLM / llama.cpp),
                                            with an OPTIONAL Anthropic arm for hard turns
    TTS  : ElevenLabs (cloud)            ->  Kokoro-82M via mlx-audio (on-device),
                                            with a macOS `say` fallback
    UI   : desktop pygame + spacebar     ->  a browser client (iPad / iPhone / desktop),
                                            push-to-talk, WebSocket audio streaming

The original author flagged "a better memory hierarchy" as the #1 wanted improvement;
this variant exposes that as a pluggable `context_provider` seam (see CONTEXT_PROVIDER) —
drop in your own RAG / memory and it is injected into the system prompt each turn.

Config (env):
  LLM_BASE_URL   OpenAI-compatible base, e.g. http://127.0.0.1:11434/v1  (Ollama)
  LLM_MODEL      model name for that server, e.g. llama3.1:8b
  LLM_API_KEY    bearer for the local server (optional; many need none)
  ANTHROPIC_API_KEY  if set, hard/complex turns route to Claude (optional ~20%)
  WHISPER_REPO   default mlx-community/whisper-large-v3-turbo
  KOKORO_VOICE   default af_heart
  CONTEXT_PROVIDER  "import.path:function" returning a str for a user utterance (optional)

Run:  python server.py <port> <token>     (needs Python >=3.10; see requirements.txt)
"""
import os, sys, json, time, base64, tempfile, subprocess, urllib.request, re, asyncio, importlib
from pathlib import Path

os.environ["PATH"] = os.path.expanduser("~/.bin") + ":/opt/homebrew/bin:" + os.environ.get("PATH", "")
HERE = Path(__file__).resolve().parent
(HERE / "tmp").mkdir(exist_ok=True)
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8600
TOKEN = sys.argv[2] if len(sys.argv) > 2 else "voiceloop"

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:11434/v1")   # Ollama default
LLM_MODEL    = os.environ.get("LLM_MODEL", "llama3.1:8b")
LLM_API_KEY  = os.environ.get("LLM_API_KEY", "")
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY")
WHISPER_REPO = os.environ.get("WHISPER_REPO", "mlx-community/whisper-large-v3-turbo")
KOKORO_VOICE = os.environ.get("KOKORO_VOICE", "af_heart")

# ---------- optional context provider (memory / RAG hook) ----------
def _load_context_provider():
    spec = os.environ.get("CONTEXT_PROVIDER", "")
    if not spec or ":" not in spec:
        return lambda user_text: ""
    mod, fn = spec.split(":", 1)
    try:
        return getattr(importlib.import_module(mod), fn)
    except Exception as e:
        print(f"[context] provider {spec!r} failed to load ({e!r}); disabled", flush=True)
        return lambda user_text: ""
context_provider = _load_context_provider()

# ---------- STT (MLX-Whisper, on-device) ----------
import mlx_whisper
def stt(audio_path: str) -> str:
    r = mlx_whisper.transcribe(audio_path, path_or_hf_repo=WHISPER_REPO)
    return (r.get("text") or "").strip()

# ---------- TTS (Kokoro via mlx-audio, `say` fallback) ----------
_KOKORO = None
def _init_tts():
    global _KOKORO
    try:
        from mlx_audio.tts.utils import load_model
        _KOKORO = load_model("prince-canuma/Kokoro-82M")
        return "kokoro-mlx"
    except Exception as e:
        print(f"[tts] Kokoro unavailable ({e!r}); using macOS `say`", flush=True)
        _KOKORO = None
        return "say"
TTS_ENGINE = None

def tts_wav(text: str) -> bytes:
    text = text.strip()
    if not text:
        return b""
    if _KOKORO is not None:
        try:
            import soundfile as sf, io, numpy as np
            out = _KOKORO.generate(text=text, voice=KOKORO_VOICE, speed=1.0)
            segs = list(out) if hasattr(out, "__iter__") and not hasattr(out, "audio") else [out]
            audio = np.concatenate([np.asarray(s.audio) for s in segs])
            buf = io.BytesIO(); sf.write(buf, audio, 24000, format="WAV", subtype="PCM_16")
            return buf.getvalue()
        except Exception as e:
            print(f"[tts] kokoro synth failed ({e!r}); say fallback", flush=True)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=HERE / "tmp") as f:
        wav = f.name
    subprocess.run(["say", "-v", "Samantha", "-o", wav, "--data-format=LEI16@16000", text], check=False)
    data = Path(wav).read_bytes(); os.unlink(wav)
    return data

# ---------- LLM (OpenAI-compatible local server; optional Claude arm) ----------
VOICE_SYS = ("You are a helpful voice assistant. Your replies are spoken aloud, so be concise, "
             "natural, and conversational — 1-3 short sentences. No markdown, no lists, no emotes.")
HARD_HINT = re.compile(r"\b(compare|why|trade[- ]?off|design|architect|explain in depth|pros and cons|analy|nuance|implication)\b", re.I)

def _openai_chat(messages, max_tokens=300):
    body = {"model": LLM_MODEL, "messages": messages, "max_tokens": max_tokens, "stream": False}
    headers = {"Content-Type": "application/json"}
    if LLM_API_KEY:
        headers["Authorization"] = f"Bearer {LLM_API_KEY}"
    req = urllib.request.Request(LLM_BASE_URL.rstrip("/") + "/chat/completions",
                                 data=json.dumps(body).encode(), headers=headers)
    r = json.load(urllib.request.urlopen(req, timeout=180))
    return r["choices"][0]["message"]["content"].strip()

def llm_reply(user_text: str, history: list):
    ctx = ""
    try:
        ctx = context_provider(user_text) or ""
    except Exception as e:
        print(f"[context] provider raised ({e!r})", flush=True)
    system = VOICE_SYS + (f"\n\nRelevant context:\n{ctx}" if ctx else "")
    # optional Claude arm for hard/complex turns
    if ANTHROPIC_KEY and HARD_HINT.search(user_text) and len(user_text.split()) > 12:
        try:
            import anthropic
            cl = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
            m = cl.messages.create(model="claude-haiku-4-5-20251001", max_tokens=300,
                                   system=system, messages=history + [{"role": "user", "content": user_text}])
            return m.content[0].text.strip(), "claude"
        except Exception as e:
            print(f"[llm] claude failed ({e!r}); local fallback", flush=True)
    msgs = [{"role": "system", "content": system}] + history[-6:] + [{"role": "user", "content": user_text}]
    return _openai_chat(msgs), f"local:{LLM_MODEL}"

def split_sentences(text: str):
    return [p for p in re.split(r"(?<=[.!?])\s+", text.strip()) if p.strip()]

# ---------- web (FastAPI + WebSocket) ----------
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, PlainTextResponse
app = FastAPI()

def _authed(request: Request) -> bool:
    return request.query_params.get("token") == TOKEN or request.cookies.get("vt") == TOKEN

@app.get("/")
def index(request: Request):
    if not _authed(request):
        return PlainTextResponse("403 - append ?token=<token>", status_code=403)
    resp = HTMLResponse((HERE / "static" / "client.html").read_text())
    resp.set_cookie("vt", TOKEN, samesite="lax")
    return resp

@app.get("/health")
def health():
    return {"ok": True, "tts": TTS_ENGINE, "claude": bool(ANTHROPIC_KEY),
            "whisper": WHISPER_REPO, "llm": LLM_MODEL}

@app.websocket("/ws")
async def ws(sock: WebSocket):
    await sock.accept()
    if sock.query_params.get("token") != TOKEN:
        await sock.close(code=4403); return
    history = []
    try:
        while True:
            msg = await sock.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if msg.get("bytes") is None:
                continue
            t0 = time.time()
            with tempfile.NamedTemporaryFile(suffix=".webm", delete=False, dir=HERE / "tmp") as f:
                f.write(msg["bytes"]); apath = f.name
            user_text = await asyncio.to_thread(stt, apath); os.unlink(apath)
            t_stt = time.time() - t0
            if not user_text:
                await sock.send_json({"type": "user_text", "text": "(no speech detected)"})
                await sock.send_json({"type": "turn_end", "timing": {"stt": round(t_stt, 2)}}); continue
            await sock.send_json({"type": "user_text", "text": user_text})
            t1 = time.time()
            reply, engine = await asyncio.to_thread(llm_reply, user_text, history)
            t_llm = time.time() - t1
            await sock.send_json({"type": "reply_text", "text": reply, "engine": engine})
            history += [{"role": "user", "content": user_text}, {"role": "assistant", "content": reply}]
            t2 = time.time(); first = None
            for sent in split_sentences(reply):
                wav = await asyncio.to_thread(tts_wav, sent)
                if first is None: first = time.time() - t2
                if wav:
                    await sock.send_json({"type": "audio", "text": sent, "b64": base64.b64encode(wav).decode()})
            await sock.send_json({"type": "turn_end", "engine": engine, "timing": {
                "stt": round(t_stt, 2), "llm": round(t_llm, 2),
                "tts_first": round(first or 0, 2), "total": round(time.time() - t0, 2)}})
    except WebSocketDisconnect:
        return

if __name__ == "__main__":
    import uvicorn
    TTS_ENGINE = _init_tts()
    print(f"[voiceloop] TTS={TTS_ENGINE} llm={LLM_MODEL} claude={bool(ANTHROPIC_KEY)} "
          f"port={PORT} token={TOKEN}", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
