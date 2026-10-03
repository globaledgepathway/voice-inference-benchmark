#!/usr/bin/env python3
"""Full voice-turn benchmark: speech-to-text (Whisper) -> streaming LLM reply.

Per turn: ASR latency, LLM time to first token, LLM time to first sentence,
voice turn = ASR + LLM time to first sentence (when the TTS could start talking).
Audio: LibriSpeech dummy clips (real speech, 2-30 s) sent as FLAC.
"""
import argparse, io, itertools, json, statistics, threading, time, urllib.request, uuid
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq

SYSTEM = ("You are a friendly phone agent for a travel company. Reply in one or two short "
          "spoken sentences. No lists, no markdown.")


def load_clips():
    path = hf_hub_download("hf-internal-testing/librispeech_asr_dummy",
                           "clean/validation-00000-of-00001.parquet", repo_type="dataset")
    t = pq.read_table(path).to_pylist()
    return [r["audio"]["bytes"] for r in t if r["audio"].get("bytes")]


def multipart(fields, fname, data):
    b = uuid.uuid4().hex
    out = io.BytesIO()
    for k, v in fields.items():
        out.write(f"--{b}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode())
    out.write(f"--{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{fname}\"\r\n"
              f"Content-Type: audio/flac\r\n\r\n".encode())
    out.write(data)
    out.write(f"\r\n--{b}--\r\n".encode())
    return out.getvalue(), f"multipart/form-data; boundary={b}"


def asr(url, model, audio):
    body, ctype = multipart({"model": model, "language": "en"}, "a.flac", audio)
    req = urllib.request.Request(url + "/audio/transcriptions", data=body,
                                 headers={"Content-Type": ctype}, method="POST")
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as r:
        text = json.load(r)["text"]
    return time.perf_counter() - t0, text


def tts(url, text):
    req = urllib.request.Request(url + "/tts", data=json.dumps({"text": text}).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as r:
        r.read()
        return time.perf_counter() - t0, float(r.headers["X-Audio-S"])


def llm(url, model, text, max_tokens):
    body = {"model": model, "stream": True, "max_tokens": max_tokens, "temperature": 0.7,
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": text}]}
    req = urllib.request.Request(url + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.perf_counter(); first = sent = None; acc = ""; sent_text = None
    with urllib.request.urlopen(req, timeout=120) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            d = line[5:].strip()
            if d == "[DONE]":
                break
            for ch in json.loads(d).get("choices", []):
                delta = (ch.get("delta") or {}).get("content")
                if delta:
                    now = time.perf_counter()
                    first = first or now
                    acc += delta
                    if sent is None and any(p in acc for p in ".!?"):
                        sent = now
                        cut = max(acc.rfind(p) for p in ".!?") + 1
                        sent_text = acc[:cut]
    end = time.perf_counter()
    return first - t0, (sent or end) - t0, end - t0, (sent_text or acc)


_rr = itertools.count(); _rr_lock = threading.Lock()


def next_tts(args):
    urls = args.tts_url.split(",")
    with _rr_lock:
        return urls[next(_rr) % len(urls)]


def turn(args, audio):
    try:
        a, text = asr(args.asr_url, args.asr_model, audio)
        ttft, tfs, e2e, sent_text = llm(args.llm_url, args.llm_model, text, args.max_tokens)
        t, audio_s = tts(next_tts(args), sent_text)
        return {"ok": True, "asr": a, "ttft": ttft, "first_sentence": tfs, "turn": a + tfs,
                "tts": t, "first_audio": a + tfs + t, "llm_e2e": e2e, "audio_s": audio_s}
    except Exception as e:
        return {"ok": False, "error": str(e)[:200]}


def pct(v, p):
    s = sorted(v); k = (len(s) - 1) * p / 100; lo = int(k); hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asr-url", default="http://localhost:8001/v1")
    ap.add_argument("--asr-model", default="openai/whisper-large-v3-turbo")
    ap.add_argument("--llm-url", default="http://localhost:8000/v1")
    ap.add_argument("--llm-model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--tts-url", default="http://localhost:8002", help="comma-separated list = round-robin over TTS workers")
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8, 16, 32])
    ap.add_argument("--turns", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=150)
    ap.add_argument("--out", default="results/voice_full_loop_summary.json")
    args = ap.parse_args()
    clips = load_clips()
    print(f"{len(clips)} clips loaded; warming up")
    for c in clips[:3]:
        turn(args, c)
    rows = []
    print("conc ok/err | ASR p50/p95 | LLM TTFT p50 | 1st sentence p50 | TTS p50/p95 | FIRST AUDIO p50/p95 (ms)")
    for c in args.concurrency:
        jobs = [clips[i % len(clips)] for i in range(max(args.turns, c * 2))]
        t0 = time.perf_counter()
        with ThreadPoolExecutor(c) as ex:
            res = list(ex.map(lambda a: turn(args, a), jobs))
        wall = time.perf_counter() - t0
        ok = [r for r in res if r["ok"]]
        if not ok:
            print(c, "all failed:", res[0].get("error")); continue
        g = lambda k: [r[k] * 1000 for r in ok]
        row = {"concurrency": c, "turns": len(res), "errors": len(res) - len(ok), "wall_s": round(wall, 2),
               "asr_p50_ms": round(pct(g("asr"), 50)), "asr_p95_ms": round(pct(g("asr"), 95)),
               "ttft_p50_ms": round(pct(g("ttft"), 50)), "ttft_p95_ms": round(pct(g("ttft"), 95)),
               "first_sentence_p50_ms": round(pct(g("first_sentence"), 50)),
               "turn_p50_ms": round(pct(g("turn"), 50)), "turn_p95_ms": round(pct(g("turn"), 95)),
               "tts_p50_ms": round(pct(g("tts"), 50)), "tts_p95_ms": round(pct(g("tts"), 95)),
               "first_audio_p50_ms": round(pct(g("first_audio"), 50)),
               "first_audio_p95_ms": round(pct(g("first_audio"), 95)),
               "first_sentence_audio_s_p50": round(pct([r["audio_s"] for r in ok], 50), 2)}
        rows.append(row)
        print(f"{c:>4} {len(ok)}/{row['errors']} | {row['asr_p50_ms']}/{row['asr_p95_ms']} | "
              f"{row['ttft_p50_ms']} | {row['first_sentence_p50_ms']} | {row['tts_p50_ms']}/{row['tts_p95_ms']} | "
              f"{row['first_audio_p50_ms']}/{row['first_audio_p95_ms']}  (spoken {row['first_sentence_audio_s_p50']}s)", flush=True)
        if len(ok) < len(res):
            print("   sample error:", next(r for r in res if not r["ok"])["error"])
    json.dump({"asr_model": args.asr_model, "llm_model": args.llm_model, "runs": rows},
              open(args.out, "w"), indent=2)
    print("saved", args.out)


if __name__ == "__main__":
    main()
