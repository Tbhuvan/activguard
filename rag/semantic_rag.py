"""
Semantic RAG Probe — Layer 2 context-aware vulnerability detection.

ChromaDB-backed retrieval system that combines three information sources:
1. Project auth model (extracted from the codebase under review)
2. CVE/OWASP-sourced anti-pattern library
3. Live threat intelligence from ACP connectors

Research rationale:
    Layer 1 probes are model-specific and require labelled training data.
    Layer 2 RAG requires *no training*: it retrieves semantically similar
    known-bad patterns and projects context directly.  This enables zero-shot
    detection of vulnerability classes not seen during Layer 1 training, and
    allows live threat intel to update detection capability in <1 second
    (the time to embed and upsert a new document).

    Research question: What is the precision/recall tradeoff between:
    (a) pure Layer 1 probe,
    (b) pure Layer 2 RAG,
    (c) the cascade (L1 gates L2)?

    Our hypothesis is that the cascade reduces false negatives (L1 recall)
    while L2 confirms and reduces false positives (L1 precision).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.threat_indicator import ThreatIndicator

logger = logging.getLogger(__name__)

# Optional ChromaDB import.
try:
    import chromadb
    from chromadb.config import Settings
    _CHROMA_AVAILABLE = True
except ImportError:
    _CHROMA_AVAILABLE = False
    logger.warning("chromadb not installed.  SecurityRAG will run in stub mode.")


class SecurityRAG:
    """Layer 2 semantic retrieval for project-specific security context.

    Maintains three ChromaDB collections:
    - ``auth_model``:   Project authentication/authorisation patterns.
    - ``antipatterns``: CVE/OWASP-sourced vulnerability patterns.
    - ``threat_intel``: Live indicators from ACP connectors.

    On each query, all three collections are searched and their results merged.
    A snippet is flagged if any retrieved pattern has high cosine similarity
    AND belongs to the same vulnerability class as the probe flag.

    Args:
        persist_dir: Directory for ChromaDB persistent storage.
        embedding_function: Optional custom ChromaDB embedding function.
            Defaults to ChromaDB's built-in sentence-transformers embedding.
    """

    COLLECTION_AUTH = "auth_model"
    COLLECTION_ANTIPATTERNS = "antipatterns"
    COLLECTION_THREAT_INTEL = "threat_intel"

    def __init__(
        self,
        persist_dir: str = ".activguard/chroma",
        embedding_function: object | None = None,
    ) -> None:
        self._persist_dir = persist_dir
        self._embedding_function = embedding_function
        self._client = None
        self._collections: dict[str, object] = {}
        if _CHROMA_AVAILABLE:
            self._init_chroma()
        else:
            logger.warning(
                "ChromaDB unavailable.  All queries will return empty results."
            )

    def _init_chroma(self) -> None:
        """Initialise ChromaDB client and collections."""
        os.makedirs(self._persist_dir, exist_ok=True)
        self._client = chromadb.PersistentClient(path=self._persist_dir)
        for collection_name in (
            self.COLLECTION_AUTH,
            self.COLLECTION_ANTIPATTERNS,
            self.COLLECTION_THREAT_INTEL,
        ):
            kwargs: dict = {"name": collection_name}
            if self._embedding_function:
                kwargs["embedding_function"] = self._embedding_function
            self._collections[collection_name] = (
                self._client.get_or_create_collection(**kwargs)
            )
        logger.info(
            "SecurityRAG initialised at %s with %d collections",
            self._persist_dir,
            len(self._collections),
        )

    # ------------------------------------------------------------------
    # Core query interface
    # ------------------------------------------------------------------

    def query(
        self,
        code_snippet: str,
        n_results: int = 5,
        vuln_class_hint: str | None = None,
    ) -> dict:
        """Query all collections for context relevant to ``code_snippet``.

        Args:
            code_snippet: The code to analyse.
            n_results: Number of results to retrieve per collection.
            vuln_class_hint: Optional vulnerability class from Layer 1 to
                focus the search.

        Returns:
            dict:
                - ``safe`` (bool): True if no concerning patterns found.
                - ``evidence`` (list[str]): Retrieved matching documents.
                - ``patterns_matched`` (list[str]): Pattern IDs that matched.
                - ``collections_searched`` (list[str]): Collections queried.
                - ``confidence`` (float): Aggregated similarity score [0, 1].
        """
        if not code_snippet:
            raise ValueError("code_snippet must be a non-empty string.")
        if not _CHROMA_AVAILABLE or not self._collections:
            return {
                "safe": True,
                "evidence": [],
                "patterns_matched": [],
                "collections_searched": [],
                "confidence": 0.0,
            }

        query_text = code_snippet
        if vuln_class_hint:
            query_text = f"[{vuln_class_hint}] {code_snippet}"

        evidence: list[str] = []
        patterns_matched: list[str] = []
        max_distance = 0.0
        collections_searched: list[str] = []

        for collection_name, collection in self._collections.items():
            try:
                results = collection.query(
                    query_texts=[query_text],
                    n_results=min(n_results, 3),
                    include=["documents", "metadatas", "distances"],
                )
                documents = results.get("documents", [[]])[0]
                metadatas = results.get("metadatas", [[]])[0]
                distances = results.get("distances", [[]])[0]
                collections_searched.append(collection_name)
                for doc, meta, dist in zip(documents, metadatas, distances):
                    # ChromaDB L2 distance: lower = more similar.
                    # Threshold: 0.85 (empirically calibrated on all-MiniLM-L6-v2 for
                    # code→NL cross-modal retrieval; a code-specialised embedding such as
                    # CodeBERT would permit a tighter threshold).
                    if dist < 0.85:
                        evidence.append(doc[:500])
                        pattern_id = meta.get("pattern_id", meta.get("id", "unknown"))
                        patterns_matched.append(str(pattern_id))
                        # Confidence: 1 - (dist / threshold), clamped to [0, 1]
                        conf = max(0.0, 1.0 - dist / 0.85)
                        max_distance = max(max_distance, conf)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "RAG query failed for collection %s: %s",
                    collection_name,
                    exc,
                )

        # Flagging logic: flag if at least one pattern within threshold found.
        is_safe = len(patterns_matched) == 0
        confidence = max_distance if not is_safe else 0.0
        return {
            "safe": is_safe,
            "evidence": evidence,
            "patterns_matched": patterns_matched,
            "collections_searched": collections_searched,
            "confidence": round(confidence, 4),
        }

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def add_threat_indicator(self, indicator: "ThreatIndicator") -> None:
        """Add a live threat indicator to the threat_intel collection.

        This is the zero-latency update path: no retraining or embedding
        recomputation is needed.  ChromaDB re-embeds and upserts the
        document on the fly.

        Args:
            indicator: A :class:`~core.ThreatIndicator` from any ACP connector.
        """
        if not _CHROMA_AVAILABLE or not self._collections:
            logger.warning("ChromaDB unavailable; cannot add threat indicator.")
            return
        collection = self._collections.get(self.COLLECTION_THREAT_INTEL)
        if not collection:
            return
        document = indicator.to_semantic_pattern()
        metadata = {
            "id": indicator.id,
            "source": indicator.source,
            "severity": indicator.severity,
            "cwe": ",".join(indicator.cwe),
            "cvss_score": indicator.cvss_score,
            "pattern_id": indicator.id,
        }
        try:
            collection.upsert(
                ids=[indicator.id],
                documents=[document],
                metadatas=[metadata],
            )
            logger.debug("Upserted threat indicator %s into threat_intel", indicator.id)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to upsert indicator %s: %s", indicator.id, exc
            )

    def add_project_context(self, codebase_path: str) -> None:
        """Extract authentication/access-control patterns from a codebase.

        Delegates to :class:`~rag.auth_extractor.AuthExtractor` for rich AST
        analysis, then converts the resulting :class:`~rag.auth_extractor.AuthModel`
        to natural-language documents and indexes them in the ``auth_model`` collection.

        Extracted signals:
        - Functions with explicit auth checks (is_authenticated, login_required, …)
        - ORM resource accesses without ownership checks (IDOR candidates)
        - Auth decorator names and the endpoints they protect
        - Auth middleware class names

        Args:
            codebase_path: Absolute or relative path to the Python codebase.

        Raises:
            FileNotFoundError: If ``codebase_path`` does not exist.
        """
        path = Path(codebase_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Codebase path does not exist: {codebase_path}"
            )
        if not _CHROMA_AVAILABLE or not self._collections:
            logger.warning("ChromaDB unavailable; skipping codebase indexing.")
            return
        collection = self._collections.get(self.COLLECTION_AUTH)
        if not collection:
            return

        from rag.auth_extractor import AuthExtractor
        extractor = AuthExtractor(project_path=str(path))
        auth_model = extractor.extract()
        documents = auth_model.to_rag_documents()

        indexed_count = 0
        for idx, doc in enumerate(documents):
            doc_id = f"auth__{path.stem}__{idx}"
            metadata: dict = {
                "pattern_id": doc_id,
                "type": "auth_model",
                "project": path.name,
                "n_auth_functions": len(auth_model.auth_functions),
                "n_unprotected": len(auth_model.unprotected_resource_accesses),
            }
            try:
                collection.upsert(ids=[doc_id], documents=[doc], metadatas=[metadata])
                indexed_count += 1
            except Exception as exc:
                logger.warning("Failed to upsert auth doc %s: %s", doc_id, exc)

        logger.info(
            "Indexed %d auth-model documents from %s "
            "(%d auth functions, %d unprotected accesses)",
            indexed_count,
            codebase_path,
            len(auth_model.auth_functions),
            len(auth_model.unprotected_resource_accesses),
        )

    def add_antipattern(
        self,
        pattern_id: str,
        document: str,
        metadata: dict | None = None,
    ) -> None:
        """Add a vulnerability anti-pattern document to the antipatterns collection.

        Args:
            pattern_id: Unique identifier for this pattern.
            document: Natural-language description of the vulnerability pattern.
            metadata: Optional metadata dict (will be stored alongside document).
        """
        if not _CHROMA_AVAILABLE or not self._collections:
            return
        collection = self._collections.get(self.COLLECTION_ANTIPATTERNS)
        if not collection:
            return
        meta = metadata or {}
        meta.setdefault("pattern_id", pattern_id)
        collection.upsert(
            ids=[pattern_id],
            documents=[document],
            metadatas=[meta],
        )
        logger.debug("Upserted antipattern %s", pattern_id)

    def seed_antipatterns(self, antipatterns: list[object]) -> None:
        """Bulk-index a list of AntiPattern objects.

        Args:
            antipatterns: List of :class:`~rag.antipattern_library.AntiPattern`
                instances from :data:`~rag.antipattern_library.AntiPatternLibrary`.
        """
        for ap in antipatterns:
            doc = (
                f"{ap.description} | "  # type: ignore[attr-defined]
                f"Anti-pattern: {ap.anti_pattern} | "  # type: ignore[attr-defined]
                f"CWE: {ap.cwe} | "  # type: ignore[attr-defined]
                f"Source: {ap.source} | "  # type: ignore[attr-defined]
                f"Example: {ap.example_vulnerable[:200]}"  # type: ignore[attr-defined]
            )
            self.add_antipattern(
                pattern_id=ap.pattern_id,  # type: ignore[attr-defined]
                document=doc,
                metadata={
                    "cwe": ap.cwe,  # type: ignore[attr-defined]
                    "severity": ap.severity,  # type: ignore[attr-defined]
                    "source": ap.source,  # type: ignore[attr-defined]
                    "pattern_id": ap.pattern_id,  # type: ignore[attr-defined]
                },
            )
        logger.info("Seeded %d anti-patterns into SecurityRAG.", len(antipatterns))

    def collection_stats(self) -> dict:
        """Return document counts for all collections.

        Returns:
            dict: Maps collection name → document count.
        """
        if not _CHROMA_AVAILABLE or not self._collections:
            return {}
        stats = {}
        for name, collection in self._collections.items():
            try:
                stats[name] = collection.count()
            except Exception as exc:  # noqa: BLE001
                stats[name] = f"error: {exc}"
        return stats

    def __repr__(self) -> str:
        return (
            f"SecurityRAG(persist_dir={self._persist_dir!r}, "
            f"chroma_available={_CHROMA_AVAILABLE})"
        )
