#!/usr/bin/env python3
"""Minimal Kokoro-82M TTS server on the GPU. POST /tts {"text": "..."} -> 24 kHz WAV.
Response headers carry synth time and audio length. One model replica; requests are
serialized on the GPU (like a single TTS worker in a voice stack)."""
import os, io, json, threading, time, wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np, torch
from kokoro import KPipeline

# MIOpen re-tunes kernels for every new sentence length on ROCm (~1.5 s each); native kernels are steady
torch.backends.cudnn.enabled = os.environ.get("TTS_MIOPEN", "0") == "1"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
pipe = KPipeline(lang_code="a", device=DEV)
VOICE = "af_heart"
lock = threading.Lock()


def synth(text):
    with lock:
        t0 = time.perf_counter()
        chunks = [a.cpu().numpy() if hasattr(a, "cpu") else np.asarray(a)
                  for _, _, a in pipe(text, voice=VOICE)]
        if DEV == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    audio = np.concatenate(chunks) if chunks else np.zeros(1, dtype=np.float32)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
    return buf.getvalue(), dt, len(audio) / 24000


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(f"ok {DEV}".encode())

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        wav, dt, dur = synth(body["text"])
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("X-Synth-Ms", f"{dt*1000:.1f}")
        self.send_header("X-Audio-S", f"{dur:.3f}")
        self.send_header("Content-Length", str(len(wav)))
        self.end_headers(); self.wfile.write(wav)


if __name__ == "__main__":
    for _ in range(3):
        synth("Warming up the voice model before we start measuring.")
    print("TTS ready on", DEV, flush=True)
    ThreadingHTTPServer(("0.0.0.0", int(os.environ.get("TTS_PORT", "8002"))), H).serve_forever()
