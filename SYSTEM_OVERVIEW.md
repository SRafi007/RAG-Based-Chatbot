# RAG-based Chatbot - System Overview

## System Architecture

```
┌───────────────────────────────────────────────────────────────────────┐
│                        RAG-BASED CHATBOT SYSTEM                        │
└───────────────────────────────────────────────────────────────────────┘

┌─────────────┐
│   FastAPI   │ ← REST API (chat, streaming, history)
└─────────────┘
      |
      v
┌─────────────────────────────────────────────────────────────────────┐
│                    LANGGRAPH ORCHESTRATOR                            │
│                                                                       │
│   ┌──────────────┐      ┌──────────────┐      ┌──────────────┐    │
│   │ DomainGuard  │ ---> │  Retriever   │ ---> │ Summarizer   │    │
│   │   Agent      │      │    Agent     │      │    Agent     │    │
│   └──────────────┘      └──────────────┘      └──────────────┘    │
│        |                       |                      |             │
│   Classifies           Hybrid RAG           Generates Response      │
│   Query Type           Retrieval                                    │
└─────────────────────────────────────────────────────────────────────┘
                                 |
                                 v
                    ┌────────────────────────┐
                    │   HYBRID RAG PIPELINE  │
                    │                        │
                    │  ┌─────────────────┐  │
                    │  │ Elasticsearch   │  │ ← Sparse (BM25)
                    │  │     BM25        │  │
                    │  └─────────────────┘  │
                    │           +            │
                    │  ┌─────────────────┐  │
                    │  │   Pinecone      │  │ ← Dense (Semantic)
                    │  │  + OpenAI Emb.  │  │
                    │  └─────────────────┘  │
                    │           |            │
                    │           v            │
                    │  ┌─────────────────┐  │
                    │  │  LLM Reranker   │  │
                    │  └─────────────────┘  │
                    └────────────────────────┘

┌──────────────────┐              ┌──────────────────┐
│  Redis (STM)     │              │ PostgreSQL (LTM) │
│  - Fast access   │              │ - Persistence    │
│  - 30-day TTL    │              │ - Analytics      │
└──────────────────┘              └──────────────────┘

┌──────────────────┐
│  Google Gemini   │ ← LLM for generation
└──────────────────┘
```

---

## Component Breakdown

### 1. API Layer (FastAPI)

**Location**: `app/api/chat_api.py`

**Endpoints**:
- `POST /api/chat/` - Standard chat
- `POST /api/chat/stream` - Streaming chat
- `GET /api/chat/history/{session_id}` - Get conversation history

**Features**:
- No authentication (uses simple user_id parameter)
- Session management
- Dual storage (Redis STM + PostgreSQL LTM)

---

### 2. Orchestrator (LangGraph)

**Location**: `app/orchestrator/orchestrator.py`

**Workflow**:
```
START → DomainGuard → [policy-related] → Retriever → Summarizer → END
                   ↘ [off-topic] → Summarizer → END
```

**State**: `app/orchestrator/state.py`
```python
AgentState {
    user_id, session_id, message,
    history, classification,
    retrieved_docs, context, reply
}
```

---

### 3. Agents

#### DomainGuard Agent
**Location**: `app/orchestrator/agents/domain_guard.py`

**Purpose**: Classify queries as "policy-related" or "off-topic"

**Output**: `state["classification"]`

#### RetrieverAgent
**Location**: `app/orchestrator/agents/retriever_agent.py`

**Purpose**: Retrieve relevant policy documents using hybrid RAG

**Features**:
- Hybrid retrieval (sparse + dense)
- LLM-based reranking
- Fallback to placeholder mode if KB not configured

**Output**: `state["retrieved_docs"]`, `state["context"]`

#### SummarizerAgent
**Location**: `app/orchestrator/agents/summarizer_agent.py`

**Purpose**: Generate final response

**Modes**:
- **Policy-related**: Uses retrieved context + conversation history
- **Off-topic**: Polite redirection to company policies

**Output**: `state["reply"]`

---

### 4. Knowledge Base Pipeline (Hybrid RAG)

**Location**: `kb_pipeline/`

#### Components

1. **Document Ingestion** (`data/ingest.py`)
   - Supports: PDF, DOCX, TXT, MD
   - Extracts text content

2. **Preprocessing** (`data/preprocess.py`)
   - Cleans text
   - Chunks documents (512 chars, 128 overlap)
   - Adds metadata

3. **Sparse Indexing** (`indexing/index_sparse.py`)
   - Elasticsearch with BM25
   - Keyword-based search
   - Fast exact matching

4. **Dense Indexing** (`indexing/index_dense.py`)
   - Pinecone with OpenAI embeddings
   - Semantic search
   - Captures meaning and context

5. **Hybrid Retriever** (`retrieval/hybrid_retriever.py`)
   - Combines sparse + dense results
   - Weighted fusion (default: 50/50)
   - Reciprocal rank fusion

6. **Reranker** (`retrieval/reranker.py`)
   - LLM-based relevance scoring
   - Improves precision
   - Optional heuristic fallback

#### Usage

**Build Index**:
```bash
python -m kb_pipeline.pipeline --mode index --data_dir data/raw
```

**Search**:
```bash
python -m kb_pipeline.pipeline --mode search --query "What is the remote work policy?" --top_k 5
```

---

### 5. Memory Management

#### Redis (Short-Term Memory)
**Location**: `app/memory/short_term_memory.py`

**Purpose**: Fast access to recent conversation history

**Features**:
- 30-day TTL per session
- Max 200 messages per session
- Automatic cleanup

**Functions**:
- `add_message(session_id, role, text)`
- `get_recent_messages(session_id, limit=20)`
- `get_session_info(session_id)`

#### PostgreSQL (Long-Term Memory)
**Location**: `app/models/conversation.py`

**Tables**:

1. **conversations**
   - Individual messages
   - Classification metadata
   - Retrieved docs data

2. **conversation_sessions**
   - Session metadata
   - Message counts
   - Timestamps

**Purpose**:
- Long-term persistence
- Analytics and reporting
- Training data collection

---

### 6. LLM Client

**Location**: `app/utils/llm_client.py`

**Provider**: Google Gemini (gemini-2.5-flash)

**Features**:
- Synchronous generation
- Async streaming support
- Error handling

**Usage**:
```python
from app.utils.llm_client import gemini_client

# Standard generation
response = gemini_client.generate("What is the policy?")

# Streaming
async for chunk in gemini_client.stream_generate("Tell me about..."):
    print(chunk, end="")
```

---

## Configuration

### Environment Variables

**.env file**:
```bash
# App
APP_NAME=RAG-based Chatbot
APP_ENV=development
APP_PORT=8000

# PostgreSQL
DATABASE_URL=postgresql://user:pass@host:port/database

# Redis
REDIS_HOST=your-redis-host
REDIS_PORT=17369
REDIS_PASSWORD=your-password
REDIS_DB=0

# Gemini
GEMINI_API_KEY=your-api-key
GEMINI_MODEL_NAME=gemini-2.5-flash

# Elasticsearch (Sparse)
ELASTIC_URL=https://your-elastic-url:9243
ELASTIC_INDEX=company_policies

# Pinecone (Dense)
PINECONE_API_KEY=your-api-key
PINECONE_ENV=your-environment
PINECONE_INDEX=company-policies

# OpenAI (Embeddings)
OPENAI_API_KEY=your-api-key
EMBEDDING_MODEL=text-embedding-3-small

# Memory
MAX_SESSION_MESSAGES=200
SESSION_TTL_DAYS=30

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/app_logs.log
```

---

## Data Flow

### User Query → Response

1. **API Request**
   ```json
   POST /api/chat/
   {
     "message": "What is the remote work policy?",
     "user_id": "user123",
     "session_id": "optional"
   }
   ```

2. **Redis STM**: Load recent conversation history (last 20 messages)

3. **DomainGuard**: Classify query → "policy-related"

4. **RetrieverAgent**:
   - Query Elasticsearch (BM25) → Get sparse results
   - Query Pinecone (semantic) → Get dense results
   - Fuse results with weights (50% sparse, 50% dense)
   - Rerank using LLM → Top 3 documents
   - Format context

5. **SummarizerAgent**:
   - Build prompt with context + history
   - Generate response using Gemini
   - Return formatted reply

6. **Storage**:
   - Save user message to Redis STM
   - Save assistant reply to Redis STM
   - Save both to PostgreSQL (conversations table)
   - Update session metadata

7. **API Response**
   ```json
   {
     "session_id": "abc123",
     "reply": "Our remote work policy allows...",
     "classification": "policy-related",
     "retrieved_docs": 3,
     "success": true
   }
   ```

---

## File Structure

```
RAG-based Chatbot/
├── app/
│   ├── api/
│   │   └── chat_api.py           # FastAPI endpoints
│   ├── orchestrator/
│   │   ├── orchestrator.py       # LangGraph workflow
│   │   ├── state.py              # State schema
│   │   └── agents/
│   │       ├── domain_guard.py   # Classification
│   │       ├── retriever_agent.py # RAG retrieval
│   │       └── summarizer_agent.py # Response generation
│   ├── models/
│   │   └── conversation.py       # PostgreSQL models
│   ├── schemas/
│   │   └── conversation.py       # Pydantic schemas
│   ├── memory/
│   │   └── short_term_memory.py  # Redis STM
│   ├── config/
│   │   ├── db.py                 # Database config
│   │   └── settings.py           # App settings
│   ├── utils/
│   │   ├── llm_client.py         # Gemini client
│   │   ├── logger.py             # Logging
│   │   └── redis_client.py       # Redis client
│   └── main.py                   # FastAPI app
├── kb_pipeline/
│   ├── data/
│   │   ├── ingest.py             # Document ingestion
│   │   └── preprocess.py         # Text chunking
│   ├── indexing/
│   │   ├── index_sparse.py       # Elasticsearch
│   │   └── index_dense.py        # Pinecone
│   ├── retrieval/
│   │   ├── hybrid_retriever.py   # Hybrid search
│   │   └── reranker.py           # LLM reranking
│   ├── pipeline.py               # Main orchestrator
│   └── README.md
├── data/
│   └── raw/                      # Policy documents (PDF, DOCX, etc.)
├── tests/
│   ├── test_llm.py               # LLM client tests
│   ├── test_new_orchestrator.py  # Orchestrator tests
│   └── redis_stm.py              # Redis STM tests
├── logs/
│   └── app_logs.log              # Application logs
├── .env                          # Environment variables
├── requirements.txt              # Python dependencies
├── SETUP.md                      # Setup guide
├── ARCHITECTURE.md               # Architecture documentation
├── KB_INTEGRATION.md             # KB pipeline integration guide
└── SYSTEM_OVERVIEW.md            # This file
```

---

## Key Features

1. **Hybrid RAG**: Combines keyword (BM25) and semantic (embeddings) search
2. **Multi-Agent System**: Modular LangGraph architecture
3. **Dual Storage**: Redis for speed, PostgreSQL for persistence
4. **Smart Classification**: DomainGuard filters off-topic queries
5. **LLM Reranking**: Improves retrieval precision
6. **Streaming Support**: Real-time response streaming
7. **Session Management**: Maintains conversation context
8. **Plug-and-Play Agents**: Easy to add new agents to workflow

---

## Quick Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run server
uvicorn app.main:app --reload

# Test LLM
python -m tests.test_llm

# Test orchestrator
python -m tests.test_new_orchestrator

# Index documents
python -m kb_pipeline.pipeline --mode index --data_dir data/raw

# Search KB
python -m kb_pipeline.pipeline --mode search --query "vacation policy" --top_k 5

# View logs
tail -f logs/app_logs.log
```

---

## Performance Characteristics

- **Latency**: ~2-5 seconds per query (with hybrid retrieval + reranking)
- **Throughput**: Depends on Gemini API rate limits
- **Scalability**: Horizontal scaling via Elasticsearch and Pinecone
- **Storage**: Minimal (vectors stored externally)
- **Cost**: Per-query cost from OpenAI embeddings + Gemini generation

---

## Security Considerations

- No authentication system (simple user_id parameter)
- API keys stored in `.env` file
- No rate limiting (add in production)
- No input sanitization (validate in production)
- PostgreSQL connection pooling enabled

---

## Monitoring and Logs

**Log Levels**:
- INFO: Normal operations
- WARNING: Fallback to placeholder mode, missing config
- ERROR: API failures, retrieval errors

**Key Log Messages**:
```
[INFO] DomainGuardAgent: Classification: policy-related
[INFO] RetrieverAgent: Retrieved 3 documents
[INFO] Hybrid retrieval returned 3 documents
[WARNING] Using placeholder retrieval
[ERROR] Hybrid retrieval failed: [error details]
```

---

## Testing

**Test Orchestrator**:
```bash
python -m tests.test_new_orchestrator
```

**Tests include**:
- Policy-related query
- Off-topic query
- Streaming response
- Multi-turn conversation

---

## Documentation

- [SETUP.md](SETUP.md) - Installation and setup
- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture
- [KB_INTEGRATION.md](KB_INTEGRATION.md) - KB pipeline integration
- [kb_pipeline/README.md](kb_pipeline/README.md) - KB pipeline details
- [SYSTEM_OVERVIEW.md](SYSTEM_OVERVIEW.md) - This file

---

## Support

For issues or questions:
1. Check relevant documentation above
2. Review logs: `logs/app_logs.log`
3. Test individual components (LLM, Redis, orchestrator)

---

## Summary

The RAG-based Chatbot is a production-ready system combining:
- **LangGraph** for workflow orchestration
- **Hybrid RAG** for superior document retrieval
- **Dual storage** for performance and persistence
- **Multi-agent architecture** for clean separation of concerns

Start by following [SETUP.md](SETUP.md), then integrate the KB pipeline using [KB_INTEGRATION.md](KB_INTEGRATION.md).
