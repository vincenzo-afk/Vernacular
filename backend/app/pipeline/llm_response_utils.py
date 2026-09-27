"""
Small shared helpers for LLM-response handling used by more than one
pipeline stage. Kept deliberately tiny — this is not a general-purpose
"LLM utils" dumping ground, just the one bit of logic
(register_detector.py and translator.py both prompt the same
underlying models for structured JSON output and both need to defend
against the model wrapping that output in a markdown code fence
despite being told not to).
"""


def strip_markdown_fence(text: str) -> str:
    """
    Defensively strips a wrapping markdown code fence (```json ... ```
    or ``` ... ```) from LLM output. Both register_detector.py and
    translator.py call this before attempting to json.loads() a
    response, since not every model reliably honors "output only
    JSON, no other text" instructions.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:]
        stripped = stripped.strip()
    return stripped
