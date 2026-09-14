#!/usr/bin/env python3
"""Local $0 Apple-Silicon variant — bidirectional streaming voice, browser client.

A fully-local, zero-API-cost variation of the two-way voice loop for Apple Silicon,
with a browser client (iPad / iPhone / desktop). Keeps the original design's spirit
(mic -> STT -> LLM -> streamed TTS -> playback) but swaps every cloud vendor for a
local model, and adds hands-free "wake word" mode + barge-in.

    STT  : faster-whisper / AssemblyAI  ->  MLX-Whisper (on-device)
    LLM  : Anthropic (cloud)            ->  any OpenAI-compatible local server
                                            (Ollama / LM Studio / vLLM / llama.cpp),
                                            optional Anthropic arm for hard turns
    TTS  : ElevenLabs (cloud)           ->  Kokoro-82M via mlx-audio (British voices too);
                                            macOS `say` fallback
    UI   : desktop pygame + spacebar    ->  browser, push-to-talk OR wake-word always-on,
                                            WebSocket audio streaming, barge-in

Features added over the base design:
  • Wake word ("Charlie") always-on mode with client-side VAD + audible turn beeps
  • Barge-in — talk over the reply (or press the button) to interrupt it
  • English-locked, hallucination-guarded STT (no wrong-language garble / repetition loops)
  • Pluggable CONTEXT_PROVIDER memory/RAG hook (the README's #1 "future angle")
  • Personal-dictionary proper-noun correction (SUBS), transcript capture (JSONL)
  • Intent-aware routing so "ideate on X" stays generative instead of fact-reciting

Config (env): LLM_BASE_URL, LLM_MODEL, LLM_API_KEY, ANTHROPIC_API_KEY, WHISPER_REPO,
KOKORO_VOICE (e.g. bm_george British male / af_heart US female), CONTEXT_PROVIDER
("module:function" returning str), WAKE_WORD (default "charlie").

Run:  python server.py <port> <token>     (Python >=3.10; see requirements.txt)
"""
import os, sys, json, time, base64, tempfile, subprocess, urllib.request, re, asyncio, importlib, datetime
from pathlib import Path

os.environ["PATH"] = os.path.expanduser("~/.bin") + ":/opt/homebrew/bin:" + os.environ.get("PATH", "")
HERE = Path(__file__).resolve().parent
(HERE / "tmp").mkdir(exist_ok=True)
TRANSCRIPTS = HERE / "transcripts"; TRANSCRIPTS.mkdir(exist_ok=True)
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8600
TOKEN = sys.argv[2] if len(sys.argv) > 2 else "voiceloop"

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:11434/v1")   # Ollama default
LLM_MODEL    = os.environ.get("LLM_MODEL", "llama3.1:8b")
LLM_API_KEY  = os.environ.get("LLM_API_KEY", "")
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_KEY")
WHISPER_REPO = os.environ.get("WHISPER_REPO", "mlx-community/whisper-large-v3-turbo")
KOKORO_VOICE = os.environ.get("KOKORO_VOICE", "bm_george")   # British male; af_heart = US female
WAKE_WORD    = os.environ.get("WAKE_WORD", "charlie")

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

# ---------- STT (MLX-Whisper, English-locked, hallucination-guarded) ----------
import mlx_whisper
# personal dictionary: fix Whisper's common mishears of YOUR proper nouns (post-transcription,
# the Wispr-Flow way; NOT an initial_prompt, which makes it hallucinate). Add your own terms.
SUBS = [(re.compile(r"\b(char lie|charley|charly|carly)\b", re.I), "Charlie")]
def _apply_subs(t: str) -> str:
    for rx, rep in SUBS: t = rx.sub(rep, t)
    return t
def _degenerate(text: str) -> bool:
    """True if Whisper hallucinated a repetition loop (e.g. 'I like a good one' x50)."""
    w = text.split()
    if len(w) < 12: return False
    grams = {" ".join(w[i:i+3]) for i in range(len(w) - 2)}
    return len(grams) <= max(3, len(w) // 8)
def stt(audio_path: str) -> str:
    # language="en" (auto-detect garbled into French/Portuguese); no initial_prompt (hallucinated);
    # no self-conditioning (kills runaway repetition); greedy decode.
    r = mlx_whisper.transcribe(audio_path, path_or_hf_repo=WHISPER_REPO,
                               language="en", condition_on_previous_text=False, temperature=0.0)
    txt = (r.get("text") or "").strip()
    if _degenerate(txt): return ""
    if len(re.sub(r"[^a-zA-Z0-9]", "", txt)) < 2: return ""   # punctuation/noise only
    return _apply_subs(txt)

# ---------- TTS (Kokoro via mlx-audio, `say` fallback) ----------
_KOKORO = None
def _init_tts():
    global _KOKORO
    try:
        from mlx_audio.tts.utils import load_model
        _KOKORO = load_model("prince-canuma/Kokoro-82M"); return "kokoro-mlx"
    except Exception as e:
        print(f"[tts] Kokoro unavailable ({e!r}); using macOS `say`", flush=True)
        _KOKORO = None; return "say"
TTS_ENGINE = None
def tts_wav(text: str) -> bytes:
    text = text.strip()
    if not text: return b""
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
    subprocess.run(["say", "-v", "Daniel", "-o", wav, "--data-format=LEI16@16000", text], check=False)
    data = Path(wav).read_bytes(); os.unlink(wav); return data

# ---------- wake word + citations ----------
WAKE_RX = re.compile(rf"\b({WAKE_WORD}|charley|charly|charli|charlee|carly|carley|carlie|karly|karlie)\b", re.I)
SLEEP_RX = re.compile(r"\b(stop listening|go to sleep|never ?mind|that'?s all|goodbye|nevermind|stand down)\b", re.I)
CITE_RX = re.compile(r"\s*\[\d+\](?:\s*\[\d+\])*")
def strip_wake(t): return WAKE_RX.sub("", t, count=1).lstrip(" ,.-—:;!?").strip()
def strip_citations(t): return CITE_RX.sub("", t).strip()   # so TTS never says "one, two"
def split_sentences(t): return [p for p in re.split(r"(?<=[.!?])\s+", t.strip()) if p.strip()]

# ---------- LLM (OpenAI-compatible local server; optional Claude arm; intent-aware) ----------
VOICE_SYS = ("You are a helpful voice assistant. Your replies are spoken aloud, so be concise, "
             "natural, and conversational — 1-3 short sentences. No markdown, no lists, no emotes.")
HARD_HINT = re.compile(r"\b(compare|why|trade[- ]?off|design|architect|explain in depth|pros and cons|analy|nuance|implication)\b", re.I)
CREATE_HINT = re.compile(r"\b(ideate|idea|ideas|brainstorm|imagine|what if|come up with|dream up|suggest|invent|write me|draft|compose|opinion|think of|creative|riff|explore)\b", re.I)
def _openai_chat(messages, max_tokens=300):
    body = {"model": LLM_MODEL, "messages": messages, "max_tokens": max_tokens, "stream": False}
    headers = {"Content-Type": "application/json"}
    if LLM_API_KEY: headers["Authorization"] = f"Bearer {LLM_API_KEY}"
    req = urllib.request.Request(LLM_BASE_URL.rstrip("/") + "/chat/completions",
                                 data=json.dumps(body).encode(), headers=headers)
    return json.load(urllib.request.urlopen(req, timeout=180))["choices"][0]["message"]["content"].strip()
def llm_reply(user_text, history):
    ctx = ""
    try: ctx = context_provider(user_text) or ""
    except Exception as e: print(f"[context] provider raised ({e!r})", flush=True)
    system = VOICE_SYS + (f"\n\nRelevant context (use if helpful):\n{ctx}" if ctx else "")
    is_hard = bool(HARD_HINT.search(user_text)) or bool(CREATE_HINT.search(user_text))
    if ANTHROPIC_KEY and is_hard and len(user_text.split()) > 12:
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

# ---------- web (FastAPI + WebSocket) ----------
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, PlainTextResponse
app = FastAPI()
def _authed(request): return request.query_params.get("token") == TOKEN or request.cookies.get("vt") == TOKEN

@app.get("/")
def index(request: Request):
    if not _authed(request): return PlainTextResponse("403 - append ?token=<token>", status_code=403)
    resp = HTMLResponse((HERE / "static" / "client.html").read_text())
    resp.set_cookie("vt", TOKEN, samesite="lax"); return resp

@app.get("/health")
def health():
    return {"ok": True, "tts": TTS_ENGINE, "claude": bool(ANTHROPIC_KEY), "whisper": WHISPER_REPO,
            "llm": LLM_MODEL, "voice": KOKORO_VOICE, "wake": WAKE_WORD}

@app.websocket("/ws")
async def ws(sock: WebSocket):
    await sock.accept()
    if sock.query_params.get("token") != TOKEN:
        await sock.close(code=4403); return
    history = []
    state = {"always_on": False, "armed": False}
    sess_file = TRANSCRIPTS / f"session-{datetime.datetime.now():%Y%m%d-%H%M%S}.jsonl"
    def log_turn(role, text, **extra):
        rec = {"ts": datetime.datetime.now().isoformat(timespec="seconds"), "role": role, "text": text, **extra}
        with open(sess_file, "a") as fh: fh.write(json.dumps(rec) + "\n")

    async def run_turn(user_text, t0, t_stt):
        try:
            await sock.send_json({"type": "user_text", "text": user_text}); log_turn("user", user_text)
            t1 = time.time()
            reply, engine = await asyncio.to_thread(llm_reply, user_text, history)
            t_llm = time.time() - t1
            await sock.send_json({"type": "reply_text", "text": reply, "engine": engine, "sources": []})
            log_turn("assistant", reply, engine=engine)
            history[:] = (history + [{"role": "user", "content": user_text},
                                     {"role": "assistant", "content": reply}])[-12:]
            t2 = time.time(); first = None
            for sent in split_sentences(strip_citations(reply)):
                wav = await asyncio.to_thread(tts_wav, sent)
                if first is None: first = time.time() - t2
                if wav: await sock.send_json({"type": "audio", "text": sent, "b64": base64.b64encode(wav).decode()})
            await sock.send_json({"type": "turn_end", "engine": engine, "timing": {
                "stt": round(t_stt, 2), "llm": round(t_llm, 2),
                "tts_first": round(first or 0, 2), "total": round(time.time() - t0, 2)}})
        except asyncio.CancelledError:
            try: await sock.send_json({"type": "interrupted"})
            except Exception: pass
            raise

    turn_task = None
    async def cancel_turn():
        nonlocal turn_task
        if turn_task and not turn_task.done():
            turn_task.cancel()
            try: await turn_task
            except (asyncio.CancelledError, Exception): pass
        turn_task = None
    try:
        while True:
            msg = await sock.receive()
            if msg.get("type") == "websocket.disconnect":
                await cancel_turn(); break
            if msg.get("text") is not None:
                try:
                    ctl = json.loads(msg["text"])
                    if ctl.get("type") == "mode":
                        await cancel_turn()
                        state["always_on"] = bool(ctl.get("always_on")); state["armed"] = False
                        await sock.send_json({"type": "mode_ack", "always_on": state["always_on"], "wake": WAKE_WORD})
                    elif ctl.get("type") == "stop":
                        await cancel_turn()
                except Exception: pass
                continue
            if msg.get("bytes") is None: continue
            await cancel_turn()                          # new audio = barge-in
            t0 = time.time()
            with tempfile.NamedTemporaryFile(suffix=".webm", delete=False, dir=HERE / "tmp") as f:
                f.write(msg["bytes"]); apath = f.name
            user_text = await asyncio.to_thread(stt, apath); os.unlink(apath)
            t_stt = time.time() - t0
            ao, armed = state["always_on"], state["armed"]
            print(f"[heard] always_on={ao} armed={armed} wake={bool(WAKE_RX.search(user_text or ''))} :: {user_text!r}", flush=True)
            if not user_text:
                if not ao:
                    await sock.send_json({"type": "user_text", "text": "(no speech detected)"})
                    await sock.send_json({"type": "turn_end", "timing": {"stt": round(t_stt, 2)}})
                continue
            if ao:
                if SLEEP_RX.search(user_text):
                    state["armed"] = False
                    await sock.send_json({"type": "wake", "state": "sleep", "text": user_text}); continue
                if not armed:
                    if WAKE_RX.search(user_text):
                        state["armed"] = True; rest = strip_wake(user_text)
                        await sock.send_json({"type": "wake", "state": "armed", "text": user_text})
                        if len(rest.split()) < 2: continue
                        user_text = rest
                    else:
                        await sock.send_json({"type": "idle_heard", "text": user_text}); continue
                else:
                    user_text = strip_wake(user_text)
            turn_task = asyncio.create_task(run_turn(user_text, t0, t_stt))
    except WebSocketDisconnect:
        await cancel_turn(); return

if __name__ == "__main__":
    import uvicorn
    TTS_ENGINE = _init_tts()
    print(f"[voiceloop] TTS={TTS_ENGINE} voice={KOKORO_VOICE} llm={LLM_MODEL} "
          f"claude={bool(ANTHROPIC_KEY)} wake={WAKE_WORD} port={PORT} token={TOKEN}", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
