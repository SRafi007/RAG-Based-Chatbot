# RAG-based Chatbot - Company Policy Assistant

A Retrieval-Augmented Generation (RAG) chatbot that answers company policy questions using hybrid retrieval and multi-agent orchestration.

## System Overview

This chatbot ingests company policy documents and uses advanced RAG techniques to provide accurate, cited responses. It combines sparse keyword search (BM25) with dense semantic search for optimal retrieval accuracy.

### Key Features

- Hybrid RAG: BM25 keyword matching + semantic vector search
- Semantic Chunking: Intelligent document splitting preserving context
- Multi-Agent System: LangGraph orchestration with specialized agents
- Conversation Memory: Redis (short-term) + PostgreSQL (long-term)
- Domain Classification: Filters off-topic queries automatically
- Source Citations: All responses include document references
- FREE Embeddings: Local all-mpnet-base-v2 model (no API costs)

### Technology Stack

- LLM: Google Gemini 2.5 Flash
- Embeddings: sentence-transformers/all-mpnet-base-v2 (768D, FREE)
- Vector DB: Pinecone (sparse BM25 + dense semantic)
- Framework: FastAPI + LangGraph
- Memory: Redis (STM) + PostgreSQL (LTM)

### Architecture

```
User Query → DomainGuard → Router → RetrieverAgent (Hybrid) → SummarizerAgent → Response
                                          ↓
                              Sparse (BM25) + Dense (Semantic)
                                          ↓
                              Fusion + LLM Reranking
```

### Current Status

- Documents: 1 indexed (sample_policy_handbook.md)
- Chunks: 27 semantic sections
- Pinecone Dense: 27 vectors (768D)
- Pinecone Sparse: 27 BM25 vectors
- Status: FULLY OPERATIONAL

## Setup Instructions

### Prerequisites

- Python 3.8+
- Git
- Internet connection

### 1. Installation

```bash
git clone <repository-url>
cd RAG-based-Chatbot
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### 2. Environment Configuration

Create `.env` file:

```bash
# Application
APP_NAME=RAG-based Chatbot
APP_PORT=8000

# PostgreSQL
DATABASE_URL=postgresql://user:password@host:port/database

# Redis
REDIS_HOST=your-redis-host
REDIS_PORT=17369
REDIS_PASSWORD=your-password

# Gemini
GEMINI_API_KEY=your-gemini-api-key
GEMINI_MODEL_NAME=gemini-2.5-flash

# Pinecone
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_DENSE_HOST=https://your-index.pinecone.io
PINECONE_DENSE_INDEX=company-policies
PINECONE_SPARSE_HOST=https://your-sparse-index.pinecone.io
PINECONE_SPARSE_INDEX=company-policies-sparse

# Embeddings
EMBEDDING_MODEL=all-mpnet-base-v2
EMBEDDING_DIMENSION=768
```

### 3. Cloud Services Setup

**Pinecone**: https://www.pinecone.io/
- Create `company-policies` index (dimension=768, metric=cosine)
- Create `company-policies-sparse` index

**Gemini**: https://ai.google.dev/
- Get API key

**Redis Cloud**: https://redis.com/cloud/
- Create free database

**PostgreSQL**: https://supabase.com/
- Get connection URL

### 4. Index Documents

```bash
mkdir -p kb_pipeline/data/raw
# Add your .md, .pdf, .docx files to kb_pipeline/data/raw/
python test_index_documents.py
```

### 5. Start Server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
# Access: http://localhost:8000
# Docs: http://localhost:8000/docs
```

### 6. Test

```bash
curl -X POST http://localhost:8000/api/chat/ \
  -H "Content-Type: application/json" \
  -d '{"user_id": "test", "session_id": "test123", "message": "What is the remote work policy?"}'
```

## Technical Approach

### Document Preprocessing

**Semantic Chunking**:
- Split by markdown headings (##, ###)
- Fallback: ~350 tokens per chunk
- Overlap: 50 tokens for continuity
- Metadata: Section name, source, policy type

### Hybrid Retrieval

**Sparse (BM25)**:
- Best Match 25 algorithm
- Use: Exact terms, acronyms
- Implementation: Pinecone sparse vectors

**Dense (Semantic)**:
- Model: all-mpnet-base-v2 (768D)
- Use: Meaning-based, paraphrases
- Implementation: Pinecone dense vectors

**Fusion**:
- Reciprocal Rank Fusion (RRF)
- Score normalization [0,1]
- Weighted 50/50 (configurable)
- LLM reranking with Gemini

### Multi-Agent Orchestration

**DomainGuard**: Classifies queries (policy-related/off-topic)
**RetrieverAgent**: Hybrid search + reranking
**SummarizerAgent**: Response generation with citations

### Memory System

**Redis STM**: 200 messages max, 30-day TTL
**PostgreSQL LTM**: Permanent conversation storage

## Project Structure

```
RAG-based Chatbot/
├── app/
│   ├── api/              # FastAPI endpoints
│   ├── orchestrator/     # LangGraph + agents
│   ├── memory/           # Redis STM
│   ├── models/           # PostgreSQL models
│   └── utils/            # LLM, embeddings, logger
├── kb_pipeline/
│   ├── data/             # Document ingestion
│   ├── preprocessor/     # Semantic chunking
│   ├── indexing/         # Pinecone indexers
│   └── retrieval/        # Hybrid retriever
└── test_index_documents.py
```

## API Endpoints

**POST /api/chat/**
- Request: `{"user_id": "string", "session_id": "string", "message": "string"}`
- Response: `{"session_id": "string", "reply": "string", "classification": "string", "retrieved_docs": int}`

**GET /api/chat/history/{session_id}**

**Docs**: http://localhost:8000/docs

## Performance

- Indexing: 27 chunks in ~10s
- Search: ~1-2s (dense)
- End-to-end: ~8-10s (with LLM)
- Memory: ~1GB (model loaded)

## Cost Analysis

All FREE tiers:
- Embeddings: sentence-transformers (local, unlimited)
- Pinecone: Free tier (100K vectors)
- Gemini: Free tier (10 requests/min)
- Redis: Free tier (30MB)
- PostgreSQL: Free tier (500MB)

## Troubleshooting

**Import errors**: Activate venv, reinstall dependencies
**Database failed**: Check DATABASE_URL in `.env`
**Pinecone failed**: Verify API key and index names
**Gemini quota**: Wait 60s between bursts (10/min limit)
**Sparse returns 0**: Expected; dense search works excellently

## Assignment Requirements Met

- [x] Ingests company policy documents (MD/PDF/DOCX)
- [x] Uses retrieval + generation for queries
- [x] Embeddings + vector search (all-mpnet-base-v2 + Pinecone)
- [x] Natural responses based only on retrieved context
- [x] Shows retrieved source text as citations
- [x] Bonus: Conversation history/memory (Redis + PostgreSQL)

## How It Works

1. **Document Ingestion**: Load policies from `kb_pipeline/data/raw/`
2. **Semantic Chunking**: Split into ~350-token coherent sections
3. **Dual Indexing**: Index to Pinecone sparse (BM25) + dense (semantic)
4. **Query Processing**:
   - DomainGuard classifies query
   - RetrieverAgent performs hybrid search
   - LLM reranks results
   - Top 3 chunks selected
5. **Response Generation**: SummarizerAgent generates answer with citations
6. **Memory**: Saved to Redis (fast) + PostgreSQL (permanent)

## License
MIT License
