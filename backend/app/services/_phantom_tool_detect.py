"""Shared phantom-tool-call detector for every chat client.

Small models (and sometimes bigger ones under stress) narrate a tool
execution in prose without emitting a real ``tool_calls`` / ``tool_use``
block:

    "Segmentation job completed. Job ID: 17087c9ebc0a563f ..."

…when nothing actually ran. This module centralizes:

  1. The pattern library that recognizes the narrative styles
     (``I have initiated``, ``stand by``, fake ``Job ID``, etc.).
  2. A one-turn ``SYSTEM`` nudge to inject into history when the pattern
     fires, telling the model to retry with a real tool_call.

Every provider client (``gemma_client``, ``cloud_llm_client``,
``claude_client``, ``gemini_client``, ``openai_client``) can import this
and reuse the same detector. That keeps the anti-hallucination rules in
one place.
"""
from __future__ import annotations

import re

# Case-insensitive regex that flags prose which describes tool execution or
# fabricates results. Matches a WIDE variety of phrasings so we catch the
# full confabulation surface without over-matching on benign short replies.
#
# Last reviewed: 2026-04-20 against NIM (Llama 3.3), Gemini 2.5, OpenAI
# GPT-5 and Gemma 4 phantom styles. The original patterns cover "I have
# initiated / please stand by / job completed / Job ID: <hex>" confabulations
# — still the dominant surface. Two 2026-era additions below catch the
# "I'm / I am running / executing / invoking" present-tense narration that
# started showing up in larger reasoning models stalling on tool plans.
# Keep this list CONSERVATIVE — the Gemma client has a tuned copy and
# over-matching legitimate speech silently erodes chat quality.
PHANTOM_TOOL_CALL_PATTERNS = re.compile(
    r"\b("
    # Announcement of tool start
    r"i\s+have\s+(initiated|started|begun|launched|kicked\s+off|triggered|run|ran)"
    r"|i'?ll\s+(notify|let\s+you\s+know|update\s+you|provide)"
    r"|please\s+stand\s+by"
    r"|please\s+wait"
    r"|the\s+process\s+(is\s+running|has\s+begun|has\s+started|has\s+completed)"
    r"|running\s+the\s+(model|inference|tool)"
    r"|inference\s+is\s+(running|now\s+in\s+progress|underway|complete)"
    r"|(process|inference|tool|segmentation|job)\s+(is\s+)?(now\s+)?(running|complete(d)?)"
    r"|i\s+will\s+(notify|update|provide\s+you|run|launch|kick\s+off)"
    r"|stand\s+by"
    r"|once\s+(the|it'?s)\s+(inference|process|tool|prediction|complete)"
    # Present-tense narration (2026-04-20 addition; reasoning-model styles).
    # Anchored to "the" / "it" / "that" to avoid matching casual "I'm
    # running late" style phrasings.
    r"|i'?m\s+(now\s+)?(running|executing|invoking)\s+(the|it|that)"
    r"|i\s+am\s+(now\s+)?(running|executing|invoking)\s+(the|it|that)"
    # Fabricated-result language
    r"|tool\s+execution\s+failed\s+on\s+the\s+backend"
    r"|endpoint\s+(for\s+)?the\s+inference\s+call\s+was\s+not\s+found"
    r"|temporary\s+issue\s+with\s+the\s+service"
    r"|(segmentation\s+)?job\s+completed"
    # Fake identifiers
    r"|job\s*id[:\s]+[a-f0-9]{6,}"
    r"|scene\s*id[:\s]+s2[abc]_"
    r")\b",
    re.IGNORECASE,
)

# System nudge to inject into history after the first phantom turn. Present
# tense + explicit "do not write another fake result" so the model is told
# both to retry AND not double-down on the confabulation.
PHANTOM_RETRY_NUDGE = (
    "PHANTOM TOOL CALL DETECTED. Your previous reply described running a "
    "tool (or invented a Job ID / Scene ID / result table) without emitting "
    "a tool_call. The tool did NOT actually run. Re-read the user's most "
    "recent request; if it needs a tool, emit the real tool_call now — do "
    "not describe the tool in prose, do not apologize, do not promise a "
    "future update, and do not write another fake result."
)


def looks_phantom(content: str) -> re.Match[str] | None:
    """Return the first matching phrase (or None) if ``content`` looks like
    a phantom tool-call narration. Thin alias over the regex for callers
    that want the match object for logging."""
    return PHANTOM_TOOL_CALL_PATTERNS.search(content or "")


# Sentinel substring that uniquely identifies a prior nudge in history.
# Exported so any callers that build a custom nudge variant still match
# the detector. DO NOT rename the prefix — it's the handshake between
# ``was_phantom_nudged`` and every client's nudge injection site.
_NUDGE_MARKER = "PHANTOM TOOL CALL DETECTED"
assert _NUDGE_MARKER in PHANTOM_RETRY_NUDGE, (
    "PHANTOM_RETRY_NUDGE must contain _NUDGE_MARKER; re-anchor the nudge "
    "or _NUDGE_MARKER before shipping."
)


def was_phantom_nudged(history: list[dict]) -> bool:
    """Has a phantom-tool retry nudge already been injected this turn?

    All four OpenAI-compatible clients (NIM, Gemini, OpenAI, local Gemma)
    cap retries at exactly ONE nudge per ``chat_with_tools`` call by
    scanning history for this sentinel. Lifting the scan into a shared
    helper means any future change to the anti-retry strategy (e.g.
    switch from content-search to a metadata marker, or bump the cap
    to 2 nudges) lands once rather than drifting across 4+ copies.

    Not called by ``claude_client`` — Claude's Messages API uses a
    different block format and rarely phantom-narrates, so the detector
    isn't wired there (by design).
    """
    return any(
        m.get("role") == "system"
        and _NUDGE_MARKER in (m.get("content") or "")
        for m in history
    )
