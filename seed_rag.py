"""
Seed Layer 2 ChromaDB with the 25 built-in anti-patterns.

Run once before using the pipeline:
    python seed_rag.py

The patterns are persisted in .activguard/chroma/ and survive across runs.
Re-running is safe (upsert is idempotent).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from rag.antipattern_library import AntiPatternLibrary
from rag.semantic_rag import SecurityRAG


def seed() -> None:
    library = AntiPatternLibrary()
    patterns = list(library._patterns.values())
    print(f"[*] Loaded {len(patterns)} anti-patterns from library")

    rag = SecurityRAG(persist_dir=".activguard/chroma")
    rag.seed_antipatterns(patterns)

    stats = rag.collection_stats()
    print(f"[+] ChromaDB seeded. Collection counts:")
    for name, count in stats.items():
        print(f"    {name}: {count} documents")

    # Quick smoke test: query for SQL injection
    result = rag.query("cursor.execute('SELECT * FROM users WHERE name = ' + username)")
    print(f"\n[*] Smoke test (SQLi query):")
    print(f"    safe={result['safe']}  confidence={result['confidence']:.3f}  patterns={result['patterns_matched']}")

    if not result["safe"]:
        print("[+] Layer 2 is live — returning real evidence.")
    else:
        print("[!] No match on smoke test (distance threshold may need tuning).")


if __name__ == "__main__":
    seed()
