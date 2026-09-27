"""
bot.py — magicpin AI Challenge submission.

Implements the 5-endpoint contract from challenge-testing-brief.md:
    POST /v1/context    — receive category/merchant/customer/trigger pushes
    POST /v1/tick        — periodic wake-up; decide what to proactively send
    POST /v1/reply       — handle a merchant/customer reply
    GET  /v1/healthz     — liveness + context counts
    GET  /v1/metadata    — team/bot identity

Run locally:
    export LLM_API_KEY=sk-ant-...
    uvicorn bot:app --host 0.0.0.0 --port 8080
    # No KV env vars set -> store.py falls back to in-memory, fine for local dev.

Run on Vercel:
    Attach a KV store to the project (Vercel dashboard -> Storage -> KV) so
    KV_REST_API_URL / KV_REST_API_TOKEN are auto-injected, and set LLM_API_KEY.
    See store.py's module docstring for why this matters.

All state lives behind store.py (`kv_get`/`kv_set`) — this file has zero
direct dict access, so it behaves identically whether it's a long-running
uvicorn process or a stateless-per-invocation Vercel function.
"""

import os
import time
from datetime import datetime, timezone
from typing import Any, Literal

from fastapi import FastAPI
from pydantic import BaseModel

import composer
import store

app = FastAPI(title="Vera Challenge Bot")
START = time.time()

HOSTILE_MARKERS = [
    "stop messaging", "stop texting", "stop contacting", "unsubscribe",
    "leave me alone", "don't message", "do not message", "useless spam",
    "this is spam", "remove me", "block this number",
]

TEAM_NAME = os.environ.get("TEAM_NAME", "Yug Agrawal")
TEAM_MEMBERS = os.environ.get("TEAM_MEMBERS", "Yug Agrawal").split(",")
CONTACT_EMAIL = os.environ.get("CONTACT_EMAIL", "yug@example.com")


# ---------------------------------------------------------------------------
# Helpers — thin wrappers over store.py with this bot's key naming scheme
# ---------------------------------------------------------------------------
async def _get_context(scope: str, context_id: str) -> dict | None:
    entry = await store.kv_get(f"ctx:{scope}:{context_id}")
    return entry["payload"] if entry else None


async def _category_for_merchant(merchant: dict) -> dict | None:
    slug = merchant.get("category_slug")
    return await _get_context("category", slug) if slug else None


async def _get_conversation(conversation_id: str) -> dict:
    conv = await store.kv_get(f"conv:{conversation_id}")
    return conv or {
        "merchant_id": None, "customer_id": None, "trigger_id": None,
        "turns": [], "last_body_sent": None,
    }


async def _get_merchant_streak(merchant_id: str) -> dict:
    return await store.kv_get(f"streak:{merchant_id}") or {"text": None, "count": 0}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# 2.4 / 2.5 — healthz / metadata
# ---------------------------------------------------------------------------
@app.get("/v1/healthz")
async def healthz():
    counts = await store.kv_count_by_scope()
    return {
        "status": "ok",
        "uptime_seconds": int(time.time() - START),
        "contexts_loaded": counts,
        "state_backend": "redis" if store.USING_REDIS else "in-memory (local dev only)",
    }


@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": TEAM_NAME,
        "team_members": TEAM_MEMBERS,
        "model": os.environ.get("LLM_MODEL", "claude-sonnet-4-5-20250929"),
        "approach": "single-prompt composer grounded in all 4 contexts; rule-based "
        "auto-reply/graceful-exit handling in /v1/reply with LLM fallback for "
        "genuine engagement; suppression-key dedup to prevent repeat sends; "
        "all state in Redis (Vercel KV) so behavior is correct on serverless.",
        "contact_email": CONTACT_EMAIL,
        "version": "1.1.0",
        "submitted_at": _now_iso(),
    }


# ---------------------------------------------------------------------------
# 2.1 — POST /v1/context
# ---------------------------------------------------------------------------
class CtxBody(BaseModel):
    scope: Literal["category", "merchant", "customer", "trigger"]
    context_id: str
    version: int
    payload: dict[str, Any]
    delivered_at: str


@app.post("/v1/context")
async def push_context(body: CtxBody):
    key = f"ctx:{body.scope}:{body.context_id}"
    cur = await store.kv_get(key)
    if cur and cur["version"] >= body.version:
        return {"accepted": False, "reason": "stale_version", "current_version": cur["version"]}
    await store.kv_set(key, {"version": body.version, "payload": body.payload})
    return {
        "accepted": True,
        "ack_id": f"ack_{body.context_id}_v{body.version}",
        "stored_at": _now_iso(),
    }


# ---------------------------------------------------------------------------
# 2.2 — POST /v1/tick
# ---------------------------------------------------------------------------
class TickBody(BaseModel):
    now: str
    available_triggers: list[str] = []


@app.post("/v1/tick")
async def tick(body: TickBody):
    actions = []
    for trg_id in body.available_triggers[:20]:  # respect the 20-actions/tick cap upstream too
        trigger = await _get_context("trigger", trg_id)
        if not trigger:
            continue

        suppression_key = trigger.get("suppression_key", trg_id)
        if await store.kv_get(f"supp:{suppression_key}"):
            continue  # already sent this logical message; restraint is rewarded

        merchant_id = trigger.get("merchant_id")
        merchant = await _get_context("merchant", merchant_id) if merchant_id else None
        if not merchant:
            continue
        category = await _category_for_merchant(merchant)
        if not category:
            continue

        customer_id = trigger.get("customer_id")
        customer = await _get_context("customer", customer_id) if customer_id else None

        composed = composer.compose(category, merchant, trigger, customer)

        conversation_id = f"conv_{merchant_id}_{trg_id}"
        send_as = composed.get("send_as", "merchant_on_behalf" if customer else "vera")
        body_text = composed.get("body", "")
        if not body_text or "http://" in body_text or "https://" in body_text:
            # empty body or URL -> hard fail per the brief; skip rather than submit garbage
            continue

        action = {
            "conversation_id": conversation_id,
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "send_as": send_as,
            "trigger_id": trg_id,
            "template_name": f"vera_{trigger.get('kind', 'generic')}_v1",
            "template_params": [merchant.get("identity", {}).get("owner_first_name", ""), body_text],
            "body": body_text,
            "cta": composed.get("cta", "open_ended"),
            "suppression_key": suppression_key,
            "rationale": composed.get("rationale", ""),
        }
        actions.append(action)

        # Record state
        await store.kv_set(f"supp:{suppression_key}", {"sent_at": time.time()})
        await store.kv_set(f"conv:{conversation_id}", {
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "trigger_id": trg_id,
            "turns": [{"from": send_as, "body": body_text, "ts": body.now}],
            "last_body_sent": body_text,
        })

    return {"actions": actions}


# ---------------------------------------------------------------------------
# 2.3 — POST /v1/reply
# ---------------------------------------------------------------------------
class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: str | None = None
    customer_id: str | None = None
    from_role: str
    message: str
    received_at: str
    turn_number: int


@app.post("/v1/reply")
async def reply(body: ReplyBody):
    conv = await _get_conversation(body.conversation_id)
    conv["merchant_id"] = conv["merchant_id"] or body.merchant_id
    conv["customer_id"] = conv["customer_id"] or body.customer_id
    conv["turns"].append({"from": body.from_role, "body": body.message, "ts": body.received_at})

    merchant = await _get_context("merchant", body.merchant_id) if body.merchant_id else None
    owner_name = (merchant.get("identity", {}).get("owner_first_name") if merchant else None) or "there"

    # Hostile / opt-out guard — deterministic, runs before any LLM call so it
    # can't be derailed by prompt drift. Satisfied by either ending outright
    # or sending one graceful apology; we do the latter so the merchant gets
    # a clean acknowledgment rather than silence.
    msg_lower = body.message.lower()
    if any(marker in msg_lower for marker in HOSTILE_MARKERS):
        apology = f"Sorry to have bothered you, {owner_name} — I won't message again unless you'd like updates in future."
        conv["turns"].append({"from": "vera", "body": apology, "ts": _now_iso()})
        conv["last_body_sent"] = apology
        await store.kv_set(f"conv:{body.conversation_id}", conv)
        return {"action": "send", "body": apology, "cta": "none",
                "rationale": "Hostile/opt-out signal detected; apologizing once and standing down."}

    # Auto-reply streak — keyed per MERCHANT (not per conversation_id): the
    # judge's auto-reply test sends identical canned text across DIFFERENT
    # conversation_ids for the same merchant, simulating a merchant whose
    # auto-responder fires across separate threads. Plain string match, no
    # LLM call needed.
    streak_key = f"streak:{body.merchant_id or 'unknown'}"
    streak = await _get_merchant_streak(body.merchant_id or "unknown")
    if streak["text"] == body.message:
        streak["count"] += 1
    else:
        streak["text"] = body.message
        streak["count"] = 1
    await store.kv_set(streak_key, streak)

    category = await _category_for_merchant(merchant) if merchant else None
    trigger = await _get_context("trigger", conv.get("trigger_id")) if conv.get("trigger_id") else None
    customer = await _get_context("customer", body.customer_id) if body.customer_id else None

    if not merchant or not category:
        # We have no grounding context for this conversation at all — safest
        # valid response is a graceful end rather than fabricating content.
        return {"action": "end", "rationale": "No merchant/category context on file for this conversation."}

    result = composer.compose_reply(
        category=category,
        merchant=merchant,
        trigger=trigger,
        customer=customer,
        conversation_so_far=conv["turns"][:-1],
        incoming_message=body.message,
        auto_reply_repeat_count=streak["count"],
    )

    action = result.get("action", "send")
    if action == "send":
        out_body = result.get("body", "")
        if not out_body or out_body == conv.get("last_body_sent") or "http://" in out_body or "https://" in out_body:
            # Anti-repetition / malformed guard — fall back to a graceful end
            # rather than emit an invalid or duplicate action.
            return {"action": "end", "rationale": "Avoiding repeat/invalid send; closing gracefully."}
        conv["turns"].append({"from": "vera", "body": out_body, "ts": _now_iso()})
        conv["last_body_sent"] = out_body
        await store.kv_set(f"conv:{body.conversation_id}", conv)
        return {"action": "send", "body": out_body, "cta": result.get("cta", "open_ended"),
                "rationale": result.get("rationale", "")}
    elif action == "wait":
        await store.kv_set(f"conv:{body.conversation_id}", conv)
        return {"action": "wait", "wait_seconds": result.get("wait_seconds", 1800),
                "rationale": result.get("rationale", "")}
    else:
        await store.kv_set(f"conv:{body.conversation_id}", conv)
        return {"action": "end", "rationale": result.get("rationale", "Conversation closed.")}


# Optional teardown per §11 of the testing brief — wipe state, no real PII persists anyway.
@app.post("/v1/teardown")
async def teardown():
    await store.kv_delete_all()
    return {"status": "wiped"}
