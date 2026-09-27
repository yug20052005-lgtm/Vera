"""
store.py — persistence layer.

Why this exists: Vercel's Python functions are serverless. A module-level
dict (`contexts = {}`) is NOT reliable state there — a cold start or a
concurrently-routed request can land on an instance that never saw an
earlier write. The challenge's /v1/context -> /v1/tick -> /v1/reply flow
depends on state written in one call being visible in a later call, so on
Vercel that state has to live somewhere external to the function instance.

Design:
- If KV_REST_API_URL / KV_REST_API_TOKEN are set (Vercel KV — auto-injected
  when you attach a KV store to your project in the Vercel dashboard) or
  UPSTASH_REDIS_REST_URL / UPSTASH_REDIS_REST_TOKEN (plain Upstash, same
  REST protocol), all state goes through Upstash's REST API.
- Otherwise (local dev, `uvicorn bot:app`), falls back to plain in-memory
  dicts — exactly what bot.py used before, so local testing/dry-runs are
  unaffected.

Only three operations are needed: get, set, and a scan-by-prefix (used once,
for /v1/healthz's per-scope context counts). Values are JSON-serialized
strings; callers pass/receive plain dicts.
"""

import json
import os
from typing import Any

import httpx

_KV_URL = os.environ.get("KV_REST_API_URL") or os.environ.get("UPSTASH_REDIS_REST_URL")
_KV_TOKEN = os.environ.get("KV_REST_API_TOKEN") or os.environ.get("UPSTASH_REDIS_REST_TOKEN")
USING_REDIS = bool(_KV_URL and _KV_TOKEN)

# ---------------------------------------------------------------------------
# In-memory fallback (local dev only — NOT safe on Vercel, see module docstring)
# ---------------------------------------------------------------------------
_mem: dict[str, str] = {}


async def _redis_cmd(*args: Any) -> Any:
    """Send one command to Upstash's REST API. `args` is the raw Redis
    command, e.g. _redis_cmd("SET", "foo", "bar") or _redis_cmd("KEYS", "ctx:*")."""
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.post(
            _KV_URL,
            headers={"Authorization": f"Bearer {_KV_TOKEN}"},
            json=list(args),
        )
        resp.raise_for_status()
        return resp.json().get("result")


async def kv_get(key: str) -> dict | None:
    if USING_REDIS:
        raw = await _redis_cmd("GET", key)
    else:
        raw = _mem.get(key)
    return json.loads(raw) if raw else None


async def kv_set(key: str, value: dict) -> None:
    raw = json.dumps(value, ensure_ascii=False)
    if USING_REDIS:
        await _redis_cmd("SET", key, raw)
    else:
        _mem[key] = raw


async def kv_delete_all() -> None:
    """Used by /v1/teardown. Only wipes keys this bot owns (our prefixes)."""
    if USING_REDIS:
        for prefix in ("ctx:", "conv:", "supp:"):
            keys = await _redis_cmd("KEYS", f"{prefix}*")
            if keys:
                await _redis_cmd("DEL", *keys)
    else:
        _mem.clear()


async def kv_count_by_scope() -> dict[str, int]:
    """Powers /v1/healthz's contexts_loaded counts."""
    counts = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
    if USING_REDIS:
        keys = await _redis_cmd("KEYS", "ctx:*")
        keys = keys or []
    else:
        keys = [k for k in _mem.keys() if k.startswith("ctx:")]
    for k in keys:
        # key shape: ctx:{scope}:{context_id}
        parts = k.split(":", 2)
        if len(parts) >= 2 and parts[1] in counts:
            counts[parts[1]] += 1
    return counts
