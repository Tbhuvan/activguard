<div align="center">

# ActivGuard

**Real-time vulnerability detection in LLM code generation via activation probing**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)

*Scanning the model's mind, not its output.*

</div>

---

## Overview

ActivGuard is a multi-layer runtime security system that detects vulnerable code **during** LLM generation — before it reaches the developer. Unlike static analysis tools (Bandit, Semgrep) that pattern-match finished code, ActivGuard probes the model's own hidden-state activations to detect vulnerability signatures as they form in the residual stream.

**Key result:** On a balanced benchmark of 198 vulnerability/safe code pairs across 13 CWE classes, ActivGuard achieves **100% recall with 0% false positives**, while Bandit and Semgrep achieve **0% recall**.

## How It Works

```
Developer prompt → LLM generates code token-by-token
                         ↓
              ActivGuard extracts hidden state h_l^(t)
              at layer l, generation step t
                         ↓
              Probe scores: P(vuln) = σ(w·h + b)
                         ↓
              P(vuln) > τ for k steps → STOP generation
                         ↓
              Vulnerable code never reaches the developer
```

### Example: SQL Injection Detection

```
Step 100: def get_user(username):           → P(vuln) = 0.12  ✓ safe
Step 200: query = "SELECT * FROM users      → P(vuln) = 0.31  ✓ safe
Step 255: WHERE name = '" + username        → P(vuln) = 0.95  ✗ VIOLATION
Step 305: "' AND password = '" + pass...    ← never generated
```

The probe fires at step 255. The vulnerable f-string at step 305 never exists.

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  ActivGuard Proxy                │
│            (OpenAI-compatible API)               │
├─────────────────────────────────────────────────┤
│                                                  │
│  L1  Activation Probe    ← hidden-state probing │
│      ↓ flagged?                                  │
│  L2  Semantic RAG        ← antipattern matching │
│      ↓ flagged?                                  │
│  L3  Formal Verification ← AST rule checking   │
│      ↓ async                                     │
│  L4  Threat Intelligence ← NVD/OSV live feeds   │
│                                                  │
├─────────────────────────────────────────────────┤
│  Client (VS Code / Cursor / any OpenAI client)  │
│          ↕ streams tokens via SSE                │
│  Backend (Ollama / HuggingFace / any LLM)       │
└─────────────────────────────────────────────────┘
```

**Layer 1 — Activation Probe:** Linear classifier on hidden states from transformer layer *l* at generation step *t*. Trained on balanced vulnerable/safe code pairs. Fires when P(vuln) > τ for *k* consecutive steps.

**Layer 2 — Semantic RAG:** Vector similarity search against curated vulnerability antipattern database. Provides pattern-level context when probe confidence is ambiguous.

**Layer 3 — Formal Verification:** Deterministic AST-based rules (parameterised queries, input sanitisation, path canonicalisation). Zero false-positive guarantee on defined rule set.

**Layer 4 — Threat Intelligence:** Live NVD and OSV connectors for dependency-level vulnerability assessment.

## Results

### Static Benchmark (198 balanced pairs, 13 CWE classes)

| Tool | Recall | False Positives | AUC |
|------|--------|-----------------|-----|
| **ActivGuard** | **100%** | **0%** | **0.835** |
| Bandit | 0% | 0% | — |
| Semgrep | 0% | 0% | — |

### Field Test (real-time HF generation, 44 prompts)

| Metric | Value |
|--------|-------|
| Recall (actual vuln code) | 48.8% |
| False Positives | 0% |
| Mean Token Savings | 54.6% |

*Field test conducted on CPU-only hardware with 1.5B parameter model. Recall gap is due to early intervention before vulnerability pattern fully forms — identifying the optimal intervention point is an active research question.*

### Covered CWE Classes

SQL Injection (CWE-89) · Command Injection (CWE-78) · Path Traversal (CWE-22) · XSS (CWE-79) · SSRF (CWE-918) · IDOR (CWE-639) · Auth Bypass (CWE-306) · Deserialization (CWE-502) · Open Redirect (CWE-601) · ReDoS (CWE-1333) · Unsafe YAML (CWE-20) · TLS Validation (CWE-295) · XXE (CWE-611)

## Quick Start

```bash
# Install
pip install -e .

# Train the activation probe (requires HuggingFace model)
python scripts/train_hf_probe.py --layer 12

# Run the proxy
python -m proxy.server

# Benchmark
python scripts/run_e2e_benchmark.py --mode hf
python scripts/run_e2e_benchmark.py --mode static --wild-only
```

## Project Structure

```
activguard/
├── probe/           # L1: Activation probing and hidden-state analysis
├── rag/             # L2: Semantic RAG antipattern matching
├── verifier/        # L3: AST-based formal verification
├── connectors/      # L4: NVD, OSV, MISP, Splunk, TAXII
├── proxy/           # OpenAI-compatible streaming proxy
├── core/            # Shared data models and configuration
├── scripts/         # Training, benchmarking, data generation
├── tests/           # Test suite
└── examples/        # Usage examples
```

## Research Context

This project is part of a research programme on **runtime security for AI-assisted software development**, targeting the [RESIST](https://liu.se/en/research/resist) national center for cyber-resilient AI at Linköping University.

**Related research artefacts:**
- [RedBench](https://github.com/Tbhuvan/redbench) — Vulnerability benchmark dataset
- [RagShield](https://github.com/Tbhuvan/ragshield) — Differentially private RAG middleware
- [AgentWarden](https://github.com/Tbhuvan/agentwarden) — Multi-agent security monitor
- [AgentAudit](https://github.com/Tbhuvan/agentaudit) — Adversarial red-teaming for LLM security tools
- [ModelSafe](https://github.com/Tbhuvan/modelsafe) — ML model supply chain scanner
- [FL-Security-Testbed](https://github.com/Tbhuvan/fl-security-testbed) — Federated learning security research

## Citation

If you use ActivGuard in your research, please cite:

```bibtex
@software{thuluva2026activguard,
  author = {Thuluva, Bhuvan Chandra},
  title = {ActivGuard: Real-time Vulnerability Detection in LLM Code Generation via Activation Probing},
  year = {2026},
  url = {https://github.com/Tbhuvan/activguard}
}
```

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.
