# Ovivo Document Intelligence Assistant

A RAG-powered document assistant for water and wastewater treatment engineering documents. Built with Streamlit, Google Gemini, and a custom vector search pipeline.

## Features

- **Semantic search** across 30+ technical PDFs (EPA guidelines, RFPs, UFGS specs, O&M manuals)
- **Diagram retrieval** — finds and displays relevant P&IDs, process flows, and engineering drawings
- **Role-based access control** — 4 user roles with document-level permissions
- **Citation tracking** — every answer cites specific pages and sections

## Architecture

- **Embeddings**: Gemini `gemini-embedding-001` (3072-dim) with cosine similarity
- **LLM**: Gemini `gemini-3.8-flash` for answer generation
- **Vision**: Gemini `gemini-3.5-flash-lite` for diagram classification and description
- **Vector search**: NumPy cosine similarity over pre-computed embeddings (16K+ chunks)
- **Frontend**: Streamlit with custom dark theme

## Demo Accounts

| Username | Password | Role | Documents |
|----------|----------|------|-----------|
| `admin` | `admin123` | Administrator | All (30+) |
| `bid_analyst` | `bid123` | Bid Analyst | 30 |
| `field_tech` | `field123` | Field Technician | 11 |
| `pfas_engineer` | `pfas123` | PFAS Sales Engineer | 6 |

## Setup

1. Clone the repo
2. Add your Gemini API key to `.streamlit/secrets.toml`:
   ```toml
   GEMINI_API_KEY = "your-key-here"
   ```
3. Install dependencies: `pip install -r requirements.txt`
4. Run: `streamlit run app.py`
