#!/usr/bin/env python3
"""Tiny fake OpenAI-compatible streaming server for dry-running the harness
without a GPU. Emits ~40 tokens at a fixed pace.

  python scripts/mock_server.py --port 8000
  # then point a target's base_url at http://localhost:8000/v1
"""
import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

WORDS = ("Sure, I can help with that. Let me check what's available and "
         "get back to you in just a moment with a couple of options that "
         "should work for your schedule next week.").split()


class Handler(BaseHTTPRequestHandler):
    ttft = 0.08
    itl = 0.01

    def log_message(self, *a):
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        max_tokens = body.get("max_tokens", 40)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        time.sleep(self.ttft)
        out = WORDS[:max_tokens]
        for w in out:
            chunk = {"choices": [{"index": 0, "delta": {"content": w + " "}}]}
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.flush()
            time.sleep(self.itl)
        usage = {"prompt_tokens": 60, "completion_tokens": len(out)}
        self.wfile.write(f"data: {json.dumps({'choices': [], 'usage': usage})}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    print(f"mock server on http://localhost:{a.port}/v1")
    ThreadingHTTPServer(("0.0.0.0", a.port), Handler).serve_forever()
