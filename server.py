"""Persistent HTTP server for nexus-1 — loads the model once and serves chat via HTTP.

Run:
    python server.py [--port 8082] [--host 127.0.0.1]

Endpoints:
    GET  /health   → {"status": "ok"}
    POST /interact → {"message": "..."} → {"response": "..."}
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

# Suppress noise before heavy imports
os.environ["TQDM_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    handlers=[logging.FileHandler(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "nexus_server.log"),
        encoding="utf-8",
    )],
)

try:
    import tqdm as _tqdm_mod
    _orig_init = _tqdm_mod.tqdm.__init__
    def _disabled_init(self, *args, **kwargs):
        kwargs["disable"] = True
        _orig_init(self, *args, **kwargs)
    _tqdm_mod.tqdm.__init__ = _disabled_init
    try:
        import tqdm.auto as _tqdm_auto
        _tqdm_auto.tqdm.__init__ = _disabled_init
    except Exception:
        pass
except Exception:
    pass

from agent import AGIAgent

_agent: AGIAgent | None = None
_agent_lock = threading.Lock()


def _get_agent() -> AGIAgent:
    global _agent
    if _agent is None:
        with _agent_lock:
            if _agent is None:
                logging.info("Initializing AGIAgent...")
                _agent = AGIAgent()
                _agent.start()
                logging.info("AGIAgent ready.")
    return _agent


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        logging.debug("HTTP " + fmt, *args)

    def _send_json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {"status": "ok"})
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/interact":
            self._send_json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            message = body.get("message", "")
        except Exception as exc:
            self._send_json(400, {"error": f"bad request: {exc}"})
            return
        try:
            agent = _get_agent()
            response = agent.interact(message)
            self._send_json(200, {"response": response})
        except Exception as exc:
            logging.exception("interact error")
            self._send_json(500, {"error": str(exc), "response": ""})


def main() -> None:
    try:
        import setproctitle
        setproctitle.setproctitle("nexus1-agent:server")
    except ImportError:
        pass

    parser = argparse.ArgumentParser(description="Nexus AGI persistent server")
    parser.add_argument("--port", type=int, default=8082)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    print(f"Loading AGI agent (this may take a moment)...", flush=True)
    _get_agent()
    print(f"Agent ready. Listening on {args.host}:{args.port}", flush=True)

    httpd = HTTPServer((args.host, args.port), _Handler)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        global _agent
        if _agent is not None:
            _agent.stop()


if __name__ == "__main__":
    main()
