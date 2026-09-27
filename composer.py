"""
composer.py — the actual "brain" of the bot.

Updated to use Groq's free OpenAI-compatible API via standard library urllib,
avoiding extra dependencies or credit card requirements.
"""

import json
import os
import urllib.request
import urllib.error

# Free Groq model (high performance, zero cost)
MODEL = os.environ.get("LLM_MODEL", "llama-3.3-70b-versatile")
API_KEY = os.environ.get("LLM_API_KEY")

# ----------------------------------------------------------------------
# System prompt — this is the bot's entire "understanding" of the task.
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
    return ctx


def compose(category: dict, merchant: dict, trigger: dict, customer: dict | None = None) -> dict:
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
    if auto_reply_repeat_count >= 3:
        return {
            "action": "end",
            "rationale": "Same message received 3x in a row — auto-reply loop with no real engagement signal. Closing to avoid wasting turns.",
        }
    if auto_reply_repeat_count == 2:
        return {
            "action": "wait",
            "wait_seconds": 86400,
            "rationale": "Same auto-reply twice in a row — owner likely not checking phone. Backing off 24h before retrying.",
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
            "respond with action=send and move straight to action-mode — do NOT ask another qualifying question.\n"
            "- If they're hostile, said 'stop', or clearly not interested, respond with action=end.\n"
            "- If they asked a genuine question or engaged normally, respond with action=send and the next best message.\n"
            "- If off-topic, politely redirect to your mission in one line, action=send.\n"
            "Reply with ONLY a JSON object: "
            '{"action": "send"|"wait"|"end", "body": "...", "cta": "...", "rationale": "..."}'
        ),
    }
    return _call_llm(user_payload, is_reply=True)


def _call_llm(user_payload: dict, is_reply: bool = False) -> dict:
    api_key = os.environ.get("LLM_API_KEY")
    if not api_key:
        return _fallback(user_payload, is_reply, error="No LLM_API_KEY set in Render Environment")

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0",
    }

    payload = {
        "model": "llama-3.3-70b-versatile",
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
    }

    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as response:
            result = json.loads(response.read().decode("utf-8"))
            text = result["choices"][0]["message"]["content"].strip()
            return json.loads(text)
    except urllib.error.HTTPError as e:
        err_msg = e.read().decode("utf-8", errors="ignore")
        return _fallback(user_payload, is_reply, error=f"Groq API {e.code}: {err_msg}")
    except Exception as e:
        return _fallback(user_payload, is_reply, error=str(e))


def _fallback(user_payload: dict, is_reply: bool, error: str | None = None) -> dict:
    merchant = user_payload.get("merchant") or {}
    name = (merchant.get("identity") or {}).get("owner_first_name", "there")
    if is_reply:
        return {
            "action": "send",
            "body": f"AI Error: {error}" if error else "AI Error: LLM_API_KEY is missing!",
            "cta": "none",
            "rationale": f"LLM composer fallback ({error}); safe holding response.",
        }
    trigger = user_payload.get("trigger") or {}
    return {
        "body": f"Hi {name}, quick update from Vera on your account.",
        "cta": "open_ended",
        "send_as": "vera",
        "suppression_key": trigger.get("suppression_key", "fallback"),
        "rationale": f"LLM composer fallback ({error}); generic fallback used.",
    }
