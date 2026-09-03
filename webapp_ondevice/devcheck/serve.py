#!/usr/bin/env python3
"""Serve a directory over http://127.0.0.1 for the Road 2 on-device demo.

`file://` is not enough for the real app: ONNX Runtime Web `fetch()`es its
.wasm runtime and the .onnx models, and cross-origin rules on `file://` block
that. This is the thin static server for those checks — correct MIME types,
no caching (files change between runs), localhost only.

    ~/playwright_venv/bin/python webapp_ondevice/devcheck/serve.py webapp_ondevice/site
    python3 webapp_ondevice/devcheck/serve.py webapp_ondevice/site --port 8123

Stdlib only, so any python3 works; the Playwright venv is not required here.
Requests are logged to stderr, which is how you see a 404 on a model file.
Stop it with Ctrl-C (or by killing the background job that started it).
"""

from __future__ import annotations

import argparse
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class DevHandler(SimpleHTTPRequestHandler):
    """Static handler with the MIME types and cache policy this demo needs."""

    # python's mimetypes already maps .wasm and .mjs on the dev machine, but the
    # table is read from the OS, so pin the ones that would break the app.
    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".wasm": "application/wasm",
        ".mjs": "text/javascript",
        ".js": "text/javascript",
        ".json": "application/json",
        ".onnx": "application/octet-stream",
    }

    def end_headers(self):
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Pragma", "no-cache")
        if self.server.cross_origin_isolated:
            # Required for SharedArrayBuffer, i.e. multi-threaded ORT-Web wasm.
            # Off by default: it also blocks cross-origin assets (a CDN copy of
            # ort-web, say) unless they send CORP headers.
            self.send_header("Cross-Origin-Opener-Policy", "same-origin")
            self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        super().end_headers()


def main() -> int:
    ap = argparse.ArgumentParser(description="Static file server for devcheck runs.")
    ap.add_argument("directory", help="directory to serve")
    ap.add_argument("--port", type=int, default=8123, help="port (default: 8123)")
    ap.add_argument("--coi", action="store_true",
                    help="send cross-origin-isolation headers (needed for threaded wasm)")
    args = ap.parse_args()

    root = Path(args.directory).expanduser().resolve()
    if not root.is_dir():
        sys.exit(f"[serve] not a directory: {root}")

    handler = partial(DevHandler, directory=str(root))
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    except OSError as exc:
        sys.exit(f"[serve] cannot bind 127.0.0.1:{args.port} — {exc}")
    server.cross_origin_isolated = args.coi

    print(f"[serve] {root}")
    print(f"[serve] http://127.0.0.1:{args.port}/"
          f"{'  (cross-origin isolated)' if args.coi else ''}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[serve] stopped")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
