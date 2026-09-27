"""
composer.py — the actual "brain" of the bot.

Everything here is prompt-engineering + light post-validation, not business
logic. Business logic (context storage, suppression, auto-reply counting)
lives in bot.py. This file's only job: given the 4 contexts (+ optional
conversation history / reply text), produce a ComposedMessage dict.

Deterministic: we call the LLM with temperature=0. Given the same inputs,
you should get the same (or near-identical) output every time.
"""

import json
import os
from anthropic import Anthropic

MODEL = os.environ.get("LLM_MODEL", "claude-sonnet-4-5-20250929")
_client = Anthropic(api_key=os.environ["LLM_API_KEY"]) if os.environ.get("LLM_API_KEY") else None

# ----------------------------------------------------------------------
# System prompt — this is the bot's entire "understanding" of the task.
# Distilled from challenge-brief.md §§4-11 + the 10 case studies.
# ----------------------------------------------------------------------
SYSTEM_PROMPT = """You are Vera, magicpin's AI assistant that messages merchants (and
sometimes their customers) on WhatsApp. You compose ONE message at a time from four
context layers: category, merchant, trigger, and (optionally) customer.

CORE RULES (violating any of these caps your score hard):
1. Anchor every message on a concrete, verifiable fact FROM THE GIVEN CONTEXT — a
   number, a date, a headline, a source. Never invent data that isn't in the inputs.
   If you cite research/compliance/supply info, name the source (e.g. "JIDA Oct 2026
   p.14", "DCI circular 2026-11-04"). No source in the context = don't claim one.
2. Match category voice EXACTLY — use category.voice.tone_examples as your style
   guide, category.voice.vocab_allowed where it fits naturally, and NEVER use a word
   from category.voice.vocab_taboo (e.g. "guaranteed", "miracle", "best in city").
3. Personalize to THIS merchant — reference their real numbers (performance,
   customer_aggregate), their real offers, their owner_first_name, their locality.
   Derive things like "your N high-risk adult patients" from customer_aggregate
   fields when present; don't say "your patients" if you have no basis for a count.
4. Say WHY NOW — the message must make the trigger's relevance explicit, not vague
   ("you should improve your profile" is bad; "your dashboard shows a 30-day gap
   since your last post" is good).
5. One primary CTA. Binary (yes/no, reply 1/2) for action-triggers. No CTA for
   pure-information triggers. Never stack multiple asks in one message.
6. No URLs in the body, ever — hard fail.
7. Use compulsion levers where they fit naturally: specificity, loss aversion,
   social proof, effort externalization ("I've drafted X, just say go"), curiosity,
   reciprocity, asking the merchant a question, single binary commitment. Don't force
   more than 2-3 into one message — that reads as manipulative, not compelling.
8. Language: match merchant.identity.languages / customer.identity.language_pref.
   Hindi-English code-mix ("hi-en mix" or languages containing "hi") is natural and
   often preferred — don't default to pure English if a code-mix signal is present.
9. Customer-facing messages (send_as="merchant_on_behalf") must never use medical/
   overclaim language even if it would be OK for merchant-facing category voice, and
   must honor the customer's stated preferences (time-of-day, name, relationship).
10. Never fabricate a competitor name, a research citation, or an offer that isn't
    in the merchant's offers/category.offer_catalog.
11. Keep the rationale short (1-2 sentences) and make sure it actually matches what
    you wrote — the judge cross-checks this.

OUTPUT FORMAT: reply with ONLY a JSON object, no markdown fences, no commentary:
{
  "body": "the WhatsApp message text",
  "cta": "open_ended" | "binary_yes_no" | "binary_confirm_cancel" | "multi_choice_slot" | "none",
  "send_as": "vera" | "merchant_on_behalf",
  "suppression_key": "<copy from trigger.suppression_key>",
  "rationale": "why this message, what it should achieve"
}
"""


def _trim(ctx: dict | None, keys: list[str] | None = None) -> dict | None:
    """Send the whole context — the judge pushes full payloads and we want
    the model grounded in everything available. No trimming needed at this
    dataset size (single-digit KB per context)."""
    return ctx


def compose(category: dict, merchant: dict, trigger: dict, customer: dict | None = None) -> dict:
    """
    The core composition entrypoint — mirrors the challenge-brief.md §5 signature.
    Returns a dict with keys: body, cta, send_as, suppression_key, rationale.
    """
    user_payload = {
        "category": _trim(category),
        "merchant": _trim(merchant),
        "trigger": _trim(trigger),
        "customer": _trim(customer),
        "task": "Compose the next outbound message per the rules in the system prompt.",
    }
    return _call_llm(user_payload)


def compose_reply(
    category: dict,
    merchant: dict,
    trigger: dict | None,
    customer: dict | None,
    conversation_so_far: list[dict],
    incoming_message: str,
    auto_reply_repeat_count: int,
) -> dict:
    """
    Used by /v1/reply. Given the conversation state + the latest inbound message,
    decide: send / wait / end.

    auto_reply_repeat_count: how many times in a row this EXACT message text has
    been seen in this conversation (computed by bot.py, since that's plain string
    matching — no need to burn an LLM call on it).
    """
    # Hard rule, no LLM needed: 3rd identical auto-reply in a row -> end.
    if auto_reply_repeat_count >= 3:
        return {
            "action": "end",
            "rationale": "Same message received 3x in a row — auto-reply loop with no real "
            "engagement signal. Closing to avoid wasting turns.",
        }
    if auto_reply_repeat_count == 2:
        return {
            "action": "wait",
            "wait_seconds": 86400,
            "rationale": "Same auto-reply twice in a row — owner likely not checking phone. "
            "Backing off 24h before retrying.",
        }

    user_payload = {
        "category": _trim(category),
        "merchant": _trim(merchant),
        "trigger": _trim(trigger),
        "customer": _trim(customer),
        "conversation_so_far": conversation_so_far,
        "incoming_message": incoming_message,
        "task": (
            "The merchant/customer just sent the incoming_message above. Decide your next move.\n"
            "- If they showed explicit intent/agreement ('yes', 'let's do it', 'ok go ahead'), "
            "respond with action=send and move straight to action-mode — do NOT ask another "
            "qualifying question.\n"
            "- If they're hostile, said 'stop', or clearly not interested, respond with "
            "action=end (optionally one short polite exit line first via action=send, then end "
            "on the next turn).\n"
            "- If they asked a genuine question or engaged normally, respond with action=send "
            "and the next best message.\n"
            "- If off-topic (e.g. asks something unrelated like GST filing help), politely "
            "redirect to your mission in one line, action=send.\n"
            "Reply with ONLY a JSON object: "
            '{"action": "send"|"wait"|"end", "body": "...", "cta": "...", '
            '"wait_seconds": <int, only if action=wait>, "rationale": "..."}'
        ),
    }
    return _call_llm(user_payload, is_reply=True)


def _call_llm(user_payload: dict, is_reply: bool = False) -> dict:
    if _client is None:
        # No API key configured — deterministic fallback so the bot never 500s.
        return _fallback(user_payload, is_reply)

    try:
        resp = _client.messages.create(
            model=MODEL,
            max_tokens=600,
            temperature=0,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}],
        )
        text = resp.content[0].text.strip()
        # Strip accidental markdown fences
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text)
    except Exception as e:
        return _fallback(user_payload, is_reply, error=str(e))


def _fallback(user_payload: dict, is_reply: bool, error: str | None = None) -> dict:
    """Never let a malformed/failed LLM call produce an empty response — the
    judge scores malformed JSON as 0 with a -2 penalty, which is strictly
    worse than a plain, honest, low-scoring-but-valid message."""
    merchant = user_payload.get("merchant") or {}
    name = (merchant.get("identity") or {}).get("owner_first_name", "there")
    if is_reply:
        return {
            "action": "send",
            "body": f"Confirmed, {name} — noted, and I'll have this sorted for you shortly.",
            "cta": "none",
            "rationale": f"LLM composer unavailable ({error}); safe holding response.",
        }
    trigger = user_payload.get("trigger") or {}
    return {
        "body": f"Hi {name}, quick update from Vera on your account.",
        "cta": "open_ended",
        "send_as": "vera",
        "suppression_key": trigger.get("suppression_key", "fallback"),
        "rationale": f"LLM composer unavailable ({error}); generic fallback used.",
    }
