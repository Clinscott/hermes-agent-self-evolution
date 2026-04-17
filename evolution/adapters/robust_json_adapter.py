"""
Robust JSON adapter that handles imperfect model output.

MiniMax-M2.7 sometimes returns malformed JSON:
- Python set literals: {20,}
- Markdown reasoning blocks: [[ ## reasoning ## ]] ... output: ...
- Partial JSON with trailing garbage

This module exports a `robust_parse` function that is monkey-patched onto
dspy.adapters.json_adapter.JSONAdapter as its `parse` method.
"""
import re
import json as _json
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from dspy.adapters.json_adapter import JSONAdapter
    from dspy.signatures.signature import Signature


def _extract_fields(text: str, output_fields: list[str]) -> dict[str, str]:
    """Extract {field: value} pairs from noisy text using multiple strategies."""
    fields: dict[str, str] = {}

    # Strip markdown code fences
    text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s*```$', '', text)

    # Strategy 1: [[ ## field ## ]] ... [[ ## next_field ## ]] blocks
    for field in output_fields:
        pattern = rf'\[\[\s*##\s*{re.escape(field)}\s*##\s*\]\]\s*\n?(.*?)(?=\[\[|```|\Z)'
        m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if m:
            content = m.group(1).strip().strip('"').strip("'")
            if content:
                fields[field] = content

    if set(fields) >= set(output_fields):
        return fields

    # Strategy 2: "field": "value" patterns in plaintext
    for field in output_fields:
        if field in fields:
            continue
        for pat in [
            rf'"{re.escape(field)}"\s*:\s*"((?:[^"\\]|\\.)*)"',
            rf"'{re.escape(field)}'\s*:\s*'((?:[^'\\]|\\.)*)'",
        ]:
            m = re.search(pat, text, re.DOTALL)
            if m:
                fields[field] = m.group(1).strip()
                break

    if set(fields) >= set(output_fields):
        return fields

    # Strategy 3: json_repair on full text
    for try_json in [_json.loads, _try_json_repair]:
        try:
            repaired = try_json(text)
            if isinstance(repaired, dict):
                for field in output_fields:
                    if field not in fields and field in repaired:
                        fields[field] = str(repaired[field])
        except Exception:
            pass

    return fields


def _try_json_repair(text: str) -> Any:
    import json_repair  # type: ignore
    return json_repair.loads(text)


def robust_parse(self, signature: "Signature", completion: str) -> dict[str, Any]:
    """
    Monkey-patched onto JSONAdapter.parse as a robust replacement.

    The `self` parameter is the JSONAdapter instance (ignored — we use the
    signature to know which output fields to extract).

    Tries to extract {reasoning, output} fields from malformed model output.
    Falls back to the original JSONAdapter.parse when recovery succeeds with
    valid JSON (including trailing garbage — the parent handles that).
    """
    output_fields = list(signature.output_fields.keys())
    fields = _extract_fields(completion, output_fields)

    if set(fields) >= set(output_fields):
        return fields

    # Fall back to original parse (needs self as first arg for bound method).
    # If the original also fails on this input, return partial fields with
    # empty strings rather than crashing the evolution run.
    try:
        return _original_json_parse(self, signature, completion)
    except Exception:
        # Original parse rejected this input (e.g. Python set literal {20,}).
        # Fill any missing fields with empty string and return what we have.
        for field in output_fields:
            if field not in fields:
                fields[field] = ""
        return fields


# Captured at import time — do NOT use super() or class methods
_original_json_parse: Any = None


def install() -> None:
    """Call this AFTER capturing the original parse but BEFORE any evolution code runs."""
    from dspy.adapters.json_adapter import JSONAdapter
    global _original_json_parse
    _original_json_parse = JSONAdapter.parse
    JSONAdapter.parse = robust_parse
