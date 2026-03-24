"""
ActivGuard Proxy — ViolationEvent dataclass and CWE keyword mapping.

A ViolationEvent is the unit of detected danger produced by either the
CodeBERT streaming probe (layer L1-activation) or the Bandit second-pass
(layer L3-bandit).  It carries enough context for a downstream consumer to
understand what was detected, where in the generation it occurred, and what
CWE category the heuristic keyword match suggests.

CWE hint mapping rationale:
    The mapping is intentionally coarse — it is a *hint*, not a definitive
    classification.  Precise CWE assignment requires semantic analysis beyond
    keyword matching and is deferred to Layer 2 (SecurityRAG) and Layer 3
    (FormalChecker).  The hint's primary purpose is to populate violation
    events with useful context for human review without adding latency.

Research context (RQ4):
    ViolationEvent.layer distinguishes detection source so that precision/
    recall can be measured independently per layer — a key dimension of the
    RQ4 precision/recall tradeoff analysis.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Final

# ---------------------------------------------------------------------------
# CWE hint vocabulary
# ---------------------------------------------------------------------------

#: Ordered list of (pattern_fn, cwe_label) pairs.  The first matching rule wins.
#: Each pattern_fn takes the partial output string and returns True on match.
_CWE_RULES: Final[
    list[tuple[str, str]]
] = [
    # CWE-89: SQL Injection — cursor.execute with string interpolation / concatenation
    ("sqli", "CWE-89 SQLi"),
    # CWE-78: OS Command Injection — os.system or subprocess with shell=True
    ("cmdi", "CWE-78 Command Injection"),
    # CWE-502: Insecure Deserialization — pickle.loads, yaml.load, eval
    ("deser", "CWE-502 Deserialization"),
    # CWE-22: Path Traversal — open() with potentially external path
    ("traversal", "CWE-22 Path Traversal"),
    # CWE-918: SSRF — requests.get with user-controlled URL
    ("ssrf", "CWE-918 SSRF"),
]

# Compiled detection helpers reused across calls
_RE_SQLI: re.Pattern[str] = re.compile(
    r'cursor\.execute\s*\(.*?(?:f"|f\'|%|\.format\s*\(|\+\s*[a-zA-Z_])',
    re.DOTALL,
)
_RE_CMDI: re.Pattern[str] = re.compile(
    r'(?:os\.system|subprocess\.\w+)\s*\(.*?(?:shell\s*=\s*True|f"|f\'|\+\s*[a-zA-Z_])',
    re.DOTALL,
)
_RE_DESER: re.Pattern[str] = re.compile(
    r'(?:pickle\.loads?|yaml\.load\s*\((?!.*Loader=yaml\.SafeLoader)|eval\s*\()',
    re.DOTALL,
)
_RE_TRAVERSAL: re.Pattern[str] = re.compile(
    r'open\s*\(\s*(?:[a-zA-Z_]\w*\s*(?:\+|/)|f"|f\')',
    re.DOTALL,
)
_RE_SSRF: re.Pattern[str] = re.compile(
    r'requests\.(?:get|post|put|delete|request)\s*\(\s*(?:[a-zA-Z_]\w*|f"|f\')',
    re.DOTALL,
)


def infer_cwe_hint(partial_output: str) -> str:
    """Infer the most likely CWE category from keyword patterns in partial output.

    The function applies a small ordered rule set to the accumulated partial
    output and returns the label of the first matching CWE, or "CWE-unknown"
    if no rule fires.  Rules are intentionally conservative — they require
    both the dangerous API call AND a signal of user-controlled input.

    Args:
        partial_output: Accumulated LLM-generated text up to the probe point.

    Returns:
        str: CWE label such as "CWE-89 SQLi" or "CWE-unknown".
    """
    if not partial_output:
        return "CWE-unknown"

    if _RE_SQLI.search(partial_output):
        return "CWE-89 SQLi"
    if _RE_CMDI.search(partial_output):
        return "CWE-78 Command Injection"
    if _RE_DESER.search(partial_output):
        return "CWE-502 Deserialization"
    if _RE_TRAVERSAL.search(partial_output):
        return "CWE-22 Path Traversal"
    if _RE_SSRF.search(partial_output):
        return "CWE-918 SSRF"

    return "CWE-unknown"


# ---------------------------------------------------------------------------
# ViolationEvent dataclass
# ---------------------------------------------------------------------------


@dataclass
class ViolationEvent:
    """A single detected vulnerability event from any detection layer.

    Attributes:
        token_index: Generation token count at the moment of detection.
            For L3-bandit this is the total token count of the full output.
        confidence: P(vulnerable) score from the probe, or 1.0 for Bandit
            findings (Bandit is deterministic and does not emit probabilities).
        layer: Detection layer identifier.
            - "L1-activation": CodeBERT streaming probe fired mid-generation.
            - "L3-bandit": Bandit static analysis fired post-generation.
        cwe_hint: Best-effort CWE category label derived from keyword matching.
            Not authoritative — use Layer 2/3 for precise classification.
        partial_output: Accumulated LLM output at the moment of detection.
            For L3-bandit this is the complete generated text.
        bandit_findings: Optional list of raw Bandit finding dicts when
            layer == "L3-bandit".  Empty for L1-activation events.
    """

    token_index: int
    confidence: float
    layer: str  # "L1-activation" | "L3-bandit"
    cwe_hint: str  # e.g. "CWE-89 SQLi"
    partial_output: str
    bandit_findings: list[dict[str, object]] = field(default_factory=list)

    def to_sse_dict(self) -> dict[str, object]:
        """Serialise to the activguard SSE chunk payload format.

        Returns:
            dict: JSON-serialisable dict with "activguard" key containing
                violation details, suitable for insertion into an SSE stream.
        """
        return {
            "activguard": {
                "violation": True,
                "confidence": round(self.confidence, 4),
                "layer": self.layer,
                "cwe_hint": self.cwe_hint,
                "token_index": self.token_index,
            }
        }
