# AML Analyst Workbench (GraphRAG + Neo4j)

A focused demo for AML/KYC analysis using a Streamlit UI, a Neo4j knowledge graph, optional SQL sources, and agent tools for Cypher generation and execution.

---

## Features

- Graph-native AML workflows: customers, accounts, transactions, alerts, PII links.
- Built-in tools: recent transactions per account, ring detection, alert status updates, memory notes.
- Streamlit UI for investigations and rapid iteration.
- Optional SQL lookups via a GenAI Toolbox MCP connector.

---

## Prerequisites

- macOS/Linux or WSL2
- Docker + Docker Compose
- Python 3.11+
- Neo4j 5.x (container provided)
- Optional: Ollama or OpenAI for text→Cypher (the code defaults to an Ollama model name)

---

## Quick Start

```bash
# 1) Clone
git clone <repo-url>
cd graphrag-kyc-agent

# 2) Python deps (fast path with uv; or use pip below)
uv venv
uv sync

# 3) Environment
cp .env.example .env
# Edit .env with:
# NEO4J_URI=bolt://localhost:7687
# NEO4J_USERNAME=neo4j
# NEO4J_PASSWORD=password
# NEO4J_DATABASE=neo4j
# Optional for LLMs:
# OPENAI_API_KEY=...
# OLLAMA_HOST=http://localhost:11434

# 4) Infra
docker-compose up -d

# 5) Seed sample dataset
python generate_kyc_dataset.py

# 6) Launch UI
streamlit run graphrag-kyc-agent/aml_workbench.py

