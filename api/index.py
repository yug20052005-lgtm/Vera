"""
Vercel entrypoint. Vercel's Python runtime auto-detects an exported `app`
ASGI object in files under /api and serves it directly — no uvicorn needed
here, Vercel's own runtime handles the ASGI protocol.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bot import app  # noqa: E402
