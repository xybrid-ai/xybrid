#!/usr/bin/env python3
"""A stdlib-only OTLP/HTTP receiver that records every export request.

    python3 capture_server.py --port 4318 --out capture.jsonl --ready-file ready.json

Accepts ``POST /v1/traces``, ``/v1/logs`` and ``/v1/metrics``, answers ``200``
with an empty JSON object (an OTLP full-success response), and appends one JSON
line per request to ``--out``. The scenario configures every binding for
OTLP/JSON (``set_json_protocol(true)`` or the binding's equivalent), so bodies
are decoded as JSON. A protobuf or undecodable body is still recorded, as
base64 with ``"encoding": "protobuf"`` or ``"invalid"``, so ``check_capture.py``
can report the misconfiguration instead of a silently empty capture.

It binds ``0.0.0.0`` by default: an Android emulator reaches the host as
``10.0.2.2`` and a device on the LAN by the host's address. Once listening, it
writes ``--ready-file`` atomically (``{"port", "host", "pid", "endpoint"}``);
``--port 0`` picks a free port, which the ready file reports. SIGTERM or SIGINT
stops it cleanly with exit status 0 after the last line is written.

Runs on python3 >= 3.9 with no third-party packages.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import signal
import sys
import threading
import time
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

SIGNALS = {"/v1/traces": "traces", "/v1/logs": "logs", "/v1/metrics": "metrics"}

# A request larger than this is refused (413) rather than buffered. The
# exporter's own batches are far smaller; this only bounds a misbehaving client.
MAX_BODY_BYTES = 64 * 1024 * 1024
# Output produced per decompression step (bounds the overshoot past the limit).
DECODE_STEP = 1 << 20


class DecodedTooLarge(Exception):
    """A compressed request that would expand past ``MAX_BODY_BYTES``."""


def bounded_decompress(raw: bytes, encoding: str) -> bytes:
    """Decompresses gzip or deflate in 1 MiB steps and stops as soon as the
    output passes ``MAX_BODY_BYTES``, so a small compressed body cannot expand
    without bound (the receiver may listen on every interface)."""
    wbits = 16 + zlib.MAX_WBITS if encoding == "gzip" else zlib.MAX_WBITS
    decoder = zlib.decompressobj(wbits)
    out = bytearray()
    pending = raw
    while True:
        out += decoder.decompress(pending, DECODE_STEP)
        if len(out) > MAX_BODY_BYTES:
            raise DecodedTooLarge()
        pending = decoder.unconsumed_tail
        if not pending:
            break
    out += decoder.flush()
    if len(out) > MAX_BODY_BYTES:
        raise DecodedTooLarge()
    if not decoder.eof:
        raise zlib.error("truncated compressed body")
    return bytes(out)


class Recorder:
    """Appends one JSON line per request; safe across handler threads."""

    def __init__(self, path: Path) -> None:
        self._lock = threading.Lock()
        self._seq = 0
        path.parent.mkdir(parents=True, exist_ok=True)
        # Truncate: one capture file describes one server run.
        self._file = path.open("w", encoding="utf-8")

    def record(self, entry: Dict[str, Any]) -> int:
        with self._lock:
            if self._file.closed:
                # A keep-alive handler thread outlived shutdown; the capture is final.
                return -1
            self._seq += 1
            entry = {"seq": self._seq, **entry}
            self._file.write(json.dumps(entry, sort_keys=True, ensure_ascii=False) + "\n")
            self._file.flush()
            return self._seq

    def close(self) -> None:
        with self._lock:
            if not self._file.closed:
                self._file.flush()
                os.fsync(self._file.fileno())
                self._file.close()


def decode_body(raw: bytes, content_type: str) -> Tuple[str, Optional[Any], Optional[str]]:
    """Returns ``(encoding, decoded JSON or None, error or None)``."""
    media = content_type.split(";", 1)[0].strip().lower()
    if media == "application/x-protobuf" or media == "application/protobuf":
        return "protobuf", None, "protobuf body; configure the binding for OTLP/JSON"
    try:
        return "json", json.loads(raw.decode("utf-8")), None
    except (UnicodeDecodeError, ValueError) as error:
        return "invalid", None, f"body is not JSON ({type(error).__name__}: {error})"


def make_handler(recorder: Recorder):
    class Handler(BaseHTTPRequestHandler):
        # HTTP/1.1 keeps the exporter's connection alive; every response below
        # therefore carries an explicit Content-Length.
        protocol_version = "HTTP/1.1"
        server_version = "xybrid-otlp-capture/1"

        def _reply(self, status: int, body: bytes, content_type: str = "application/json") -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_body(self) -> Optional[bytes]:
            if "chunked" in self.headers.get("Transfer-Encoding", "").lower():
                chunks = []
                total = 0
                while True:
                    size = int(self.rfile.readline().split(b";", 1)[0].strip() or b"0", 16)
                    if size == 0:
                        # Discard trailers up to the blank line.
                        while self.rfile.readline() not in (b"\r\n", b"\n", b""):
                            pass
                        break
                    total += size
                    if total > MAX_BODY_BYTES:
                        return None
                    chunks.append(self.rfile.read(size))
                    self.rfile.readline()
                return b"".join(chunks)
            length = int(self.headers.get("Content-Length", "0") or "0")
            if length > MAX_BODY_BYTES:
                return None
            return self.rfile.read(length) if length > 0 else b""

        def do_POST(self) -> None:  # noqa: N802 -- http.server API
            path = self.path.split("?", 1)[0].rstrip("/") or "/"
            raw = self._read_body()
            if raw is None:
                self.close_connection = True
                self._reply(413, b'{"message":"body too large"}')
                return
            content_type = self.headers.get("Content-Type", "")
            content_encoding = self.headers.get("Content-Encoding", "").strip().lower()
            entry: Dict[str, Any] = {
                "received_unix_nano": time.time_ns(),
                "method": "POST",
                "path": path,
                "signal": SIGNALS.get(path),
                "content_type": content_type,
                "content_encoding": content_encoding or None,
                "user_agent": self.headers.get("User-Agent"),
                "size": len(raw),
            }
            try:
                if content_encoding in ("gzip", "deflate"):
                    raw = bounded_decompress(raw, content_encoding)
            except DecodedTooLarge:
                entry.update(encoding="invalid", body=None,
                             error=f"decoded {content_encoding} body exceeds {MAX_BODY_BYTES} bytes")
                recorder.record(entry)
                self._reply(413, b'{"message":"decoded request body too large"}')
                return
            except (OSError, EOFError, zlib.error) as error:
                entry.update(encoding="invalid", body=None, error=f"cannot decode {content_encoding}: {error}")
                entry["body_base64"] = base64.b64encode(raw).decode("ascii")
                recorder.record(entry)
                self._reply(400, b'{"message":"undecodable content-encoding"}')
                return
            encoding, body, error = decode_body(raw, content_type)
            entry.update(encoding=encoding, body=body, error=error)
            if body is None:
                entry["body_base64"] = base64.b64encode(raw).decode("ascii")
            recorder.record(entry)
            if entry["signal"] is None:
                self._reply(404, b'{"message":"not an OTLP signal path"}')
                return
            # An empty JSON object is a full-success Export*ServiceResponse.
            self._reply(200, b"{}")

        def do_GET(self) -> None:  # noqa: N802 -- http.server API
            # A liveness probe for drivers that cannot read the ready file
            # (for example a device checking it can reach the host).
            if self.path.split("?", 1)[0] in ("/", "/healthz"):
                self._reply(200, b'{"status":"ok"}')
            else:
                self._reply(404, b'{"message":"not found"}')

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 -- http.server API
            pass

    return Handler


def write_ready_file(path: Path, payload: Dict[str, Any]) -> None:
    """Writes ``payload`` so a reader never sees a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--host", default="0.0.0.0", help="bind address (default 0.0.0.0)")
    parser.add_argument("--port", type=int, default=4318, help="bind port; 0 picks a free one (default 4318)")
    parser.add_argument("--out", type=Path, required=True, help="JSONL file to write, truncated at start")
    parser.add_argument("--ready-file", type=Path, help="written once listening: port, host, pid, endpoint")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    recorder = Recorder(args.out)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(recorder))
    server.daemon_threads = True
    port = server.server_address[1]
    stop = threading.Event()

    def request_stop(signum, frame):  # noqa: ARG001 -- signal handler signature
        stop.set()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    serving = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
    serving.start()
    local = "127.0.0.1" if args.host in ("0.0.0.0", "") else args.host
    ready = {"host": args.host, "port": port, "pid": os.getpid(), "endpoint": f"http://{local}:{port}"}
    if args.ready_file is not None:
        write_ready_file(args.ready_file, ready)
    print(json.dumps(ready, sort_keys=True), flush=True)
    try:
        # A timed wait keeps the main thread responsive to signals everywhere.
        while not stop.wait(0.2):
            pass
    finally:
        server.shutdown()
        server.server_close()
        recorder.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
