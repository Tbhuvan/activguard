"""
ActivGuard Layer 2 — Semantic RAG Probe.

Retrieves project-specific security context and known vulnerability patterns
before flagging LLM-generated code.  Three ChromaDB collections are maintained:

- ``auth_model``:    project-specific authentication and access-control context.
- ``antipatterns``:  CVE/OWASP-sourced vulnerability patterns.
- ``threat_intel``:  live indicators ingested from ACP connectors.
"""

from .semantic_rag import SecurityRAG
from .antipattern_library import AntiPattern, AntiPatternLibrary
from .stix_encoder import STIXEncoder

__all__ = ["SecurityRAG", "AntiPattern", "AntiPatternLibrary", "STIXEncoder"]
