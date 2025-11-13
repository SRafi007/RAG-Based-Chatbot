# RAG-based Chatbot with Hybrid Retrieval

A production-ready chatbot system for company policy queries using hybrid RAG (Retrieval-Augmented Generation) with LangGraph orchestration.

---

## 🌟 Features

- **Hybrid RAG**: Combines sparse (BM25) and dense (semantic) retrieval for superior accuracy
- **Semantic Chunking**: Preserves document structure with heading-based splitting
- **Multi-Agent System**: LangGraph orchestration with plug-and-play architecture
- **Dual Storage**: Redis for fast access, PostgreSQL for persistence
- **Smart Classification**: Filters off-topic queries automatically
- **Streaming Support**: Real-time response generation
- **Rich Metadata**: Full attribution with section names and sources

---

## 🏗️ Architecture

```
User Query
    ↓
[DomainGuard] → Classify: policy-related or off-topic
    ↓
[RetrieverAgent] → Hybrid RAG (Elasticsearch BM25 + Pinecone Semantic)
    ↓
[LLM Reranker] → Improve precision
    ↓
[SummarizerAgent] → Generate response with context
    ↓
Response with citations
```

**Technology Stack**:
- **LLM**: Google Gemini 2.5 Flash
- **Orchestration**: LangGraph (state-based workflow)
- **Sparse Retrieval**: Elasticsearch with BM25
- **Dense Retrieval**: Pinecone with OpenAI embeddings
- **API**: FastAPI
- **STM**: Redis
- **LTM**: PostgreSQL

---

## 📋 Documentation

### Getting Started
- 🚀 **[CLOUD_SETUP_GUIDE.md](CLOUD_SETUP_GUIDE.md)** - Complete guide for setting up Elasticsearch, Pinecone, and OpenAI (for beginners)
- 📋 **[QUICK_SETUP_REFERENCE.md](QUICK_SETUP_REFERENCE.md)** - Quick reference card with commands and troubleshooting

### Technical Documentation
- 🏛️ **[ARCHITECTURE.md](ARCHITECTURE.md)** - System architecture and design decisions
- 📊 **[SYSTEM_OVERVIEW.md](SYSTEM_OVERVIEW.md)** - Component breakdown and data flow
- 🔗 **[KB_INTEGRATION.md](KB_INTEGRATION.md)** - Knowledge base pipeline integration guide
- 📚 **[KB_SEMANTIC_CHUNKING.md](KB_SEMANTIC_CHUNKING.md)** - Semantic chunking strategy explained

### Pipeline Documentation
- 📖 **[kb_pipeline/README.md](kb_pipeline/README.md)** - KB pipeline detailed documentation

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Redis (cloud or local)
- PostgreSQL (Supabase recommended)
- Google Gemini API key

### 1. Installation

```bash
# Clone repository
git clone <your-repo-url>
cd RAG-based-Chatbot

# Create virtual environment
python -m venv rag_env

# Activate virtual environment
# Windows:
rag_env\Scripts\activate
# Linux/Mac:
source rag_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

Create `.env` file:

```bash
# App
APP_NAME=RAG-based Chatbot
APP_ENV=development
APP_PORT=8000

# PostgreSQL
DATABASE_URL=postgresql://user:password@host:port/database

# Redis
REDIS_HOST=your-redis-host
REDIS_PORT=17369
REDIS_PASSWORD=your-password
REDIS_DB=0

# Google Gemini
GEMINI_API_KEY=your-gemini-api-key
GEMINI_MODEL_NAME=gemini-2.5-flash

# Elasticsearch (optional - for hybrid RAG)
ELASTIC_URL=https://your-deployment.es.us-east-1.aws.found.io:9243
ELASTIC_INDEX=company_policies

# Pinecone (optional - for hybrid RAG)
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_ENV=us-east-1
PINECONE_INDEX=company-policies

# OpenAI (optional - for embeddings)
OPENAI_API_KEY=your-openai-api-key
EMBEDDING_MODEL=text-embedding-3-small

# Memory
MAX_SESSION_MESSAGES=200
SESSION_TTL_DAYS=30

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/app_logs.log
```

### 3. Set Up Cloud Services (Optional)

**For hybrid RAG functionality**, set up Elasticsearch and Pinecone:

📖 **Follow**: [CLOUD_SETUP_GUIDE.md](CLOUD_SETUP_GUIDE.md) for step-by-step instructions

Or use the quick reference: [QUICK_SETUP_REFERENCE.md](QUICK_SETUP_REFERENCE.md)

**Test connections**:
```bash
python test_cloud_connections.py
```

### 4. Index Documents (Optional)

**For hybrid RAG with your own documents**:

```bash
# Add documents to the data folder
mkdir -p kb_pipeline/data/raw
# Copy your PDF, DOCX, TXT, MD files here

# Install KB dependencies
pip install elasticsearch pinecone-client openai PyPDF2 python-docx

# Run indexing
python -m kb_pipeline.pipeline --mode index --data_dir kb_pipeline/data/raw

# Test search
python -m kb_pipeline.pipeline --mode search --query "What is the remote work policy?"
```

### 5. Run the Application

```bash
# Start server
uvicorn app.main:app --reload --port 8000

# Server will be available at http://localhost:8000
```

### 6. Test the System

```bash
# Test orchestrator
python -m tests.test_new_orchestrator

# Test LLM client
python -m tests.test_llm
```

---

## 📡 API Usage

### Standard Chat

```bash
curl -X POST http://localhost:8000/api/chat/ \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is the remote work policy?",
    "user_id": "user123"
  }'
```

**Response**:
```json
{
  "session_id": "abc123",
  "reply": "Our remote work policy allows employees to work remotely up to 3 days per week...",
  "classification": "policy-related",
  "retrieved_docs": 3,
  "success": true
}
```

### Streaming Chat

```bash
curl -X POST http://localhost:8000/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Tell me about vacation policies",
    "user_id": "user123"
  }'
```

### Get History

```bash
curl http://localhost:8000/api/chat/history/{session_id}
```

### Interactive API Docs

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## 🧪 Testing

### Test Semantic Chunking

```bash
python test_semantic_chunking.py
```

### Test Cloud Connections

```bash
python test_cloud_connections.py
```

### Test Orchestrator

```bash
python -m tests.test_new_orchestrator
```

**Tests include**:
- Policy-related query
- Off-topic query
- Streaming response
- Multi-turn conversation

---

## 🎯 Usage Modes

### Mode 1: Placeholder (Default)

Uses hardcoded example policies - good for testing without cloud setup:

```python
# app/orchestrator/agents/retriever_agent.py
retriever_agent = RetrieverAgent(use_hybrid=False)
```

### Mode 2: Hybrid RAG (Production)

Uses Elasticsearch + Pinecone for real document retrieval:

```python
# After indexing your documents
retriever_agent = RetrieverAgent(use_hybrid=True)
```

---

## 📊 System Components

### 1. API Layer (FastAPI)
- **Location**: `app/api/chat_api.py`
- **Endpoints**: `/chat/`, `/chat/stream`, `/chat/history/{session_id}`

### 2. Orchestrator (LangGraph)
- **Location**: `app/orchestrator/orchestrator.py`
- **State**: `app/orchestrator/state.py`
- **Workflow**: START → DomainGuard → Retriever → Summarizer → END

### 3. Agents
- **DomainGuard**: Classifies queries as policy-related or off-topic
- **RetrieverAgent**: Hybrid RAG retrieval (BM25 + semantic)
- **SummarizerAgent**: Generates responses with context

### 4. Knowledge Base Pipeline
- **Ingestion**: `kb_pipeline/preprocessor/ingest.py`
- **Preprocessing**: `kb_pipeline/preprocessor/preprocess.py`
- **Sparse Indexing**: `kb_pipeline/indexing/index_sparse.py`
- **Dense Indexing**: `kb_pipeline/indexing/index_dense.py`
- **Hybrid Retrieval**: `kb_pipeline/retrieval/hybrid_retriever.py`
- **Reranker**: `kb_pipeline/retrieval/reranker.py`

### 5. Memory Management
- **Redis STM**: `app/memory/short_term_memory.py` (30-day TTL, 200 messages max)
- **PostgreSQL LTM**: `app/models/conversation.py` (permanent storage)

---

## 🔧 Configuration

### Chunk Size (Semantic)

Edit `kb_pipeline/preprocessor/preprocess.py`:

```python
DocumentPreprocessor(
    target_tokens=350,      # Target 300-400 tokens
    max_tokens=450,         # Force split if exceeded
    overlap_tokens=50,      # Continuity overlap
    min_tokens=50           # Minimum valid chunk
)
```

### Retrieval Weights

```python
# More keyword-focused
retriever = HybridRetriever(sparse_weight=0.7, dense_weight=0.3)

# More semantic-focused
retriever = HybridRetriever(sparse_weight=0.3, dense_weight=0.7)

# Balanced (default)
retriever = HybridRetriever(sparse_weight=0.5, dense_weight=0.5)
```

### Reranking

```python
# LLM reranking (higher quality)
reranker = Reranker(use_llm=True)

# Heuristic reranking (faster, cheaper)
reranker = Reranker(use_llm=False)
```

---

## 📁 Project Structure

```
RAG-based Chatbot/
├── app/
│   ├── api/                    # FastAPI endpoints
│   ├── orchestrator/           # LangGraph workflow
│   │   ├── agents/             # DomainGuard, Retriever, Summarizer
│   │   ├── orchestrator.py     # Main workflow
│   │   └── state.py            # State schema
│   ├── models/                 # PostgreSQL models
│   ├── schemas/                # Pydantic schemas
│   ├── memory/                 # Redis STM
│   ├── config/                 # Settings & DB
│   ├── utils/                  # LLM client, logger, Redis
│   └── main.py                 # FastAPI app
├── kb_pipeline/
│   ├── preprocessor/           # Document ingestion & chunking
│   ├── indexing/               # Elasticsearch & Pinecone
│   ├── retrieval/              # Hybrid retriever & reranker
│   ├── pipeline.py             # Main orchestrator
│   └── README.md
├── tests/                      # Test files
├── logs/                       # Application logs
├── .env                        # Environment variables
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## 💰 Cost Estimates

| Service | Free Tier | Usage | Monthly Cost |
|---------|-----------|-------|--------------|
| **Pinecone** | 100K vectors | 10K vectors | Free |
| **Elasticsearch** | 14-day trial | N/A | $50-100 (smallest) |
| **OpenAI** | $5 credits | ~10K chunks | ~$0.10 |
| **Gemini** | Free tier | API calls | Free |

**Tips to minimize costs**:
- Use free tiers for testing
- Delete cloud resources when not using
- Set strict usage limits
- Consider local alternatives (see below)

---

## 🔄 Alternatives (Cost-Saving)

### Free/Local Alternatives

| Component | Current | Alternative |
|-----------|---------|-------------|
| **Elasticsearch** | Cloud ($50-100/mo) | Local Elasticsearch (free) |
| **Pinecone** | Cloud ($70/mo) | Qdrant, Weaviate, ChromaDB (free) |
| **OpenAI Embeddings** | API ($0.13/1M tokens) | Sentence Transformers (free) |

---

## 🐛 Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| Import errors | Activate virtual environment, install dependencies |
| Database connection failed | Check DATABASE_URL in `.env` |
| Redis connection failed | Verify Redis host, port, password |
| Elasticsearch auth failed | Check URL includes `https://` and `:9243` |
| Pinecone index not found | Create index in Pinecone dashboard |
| OpenAI quota exceeded | Add payment method, check usage limits |

### Check Logs

```bash
# View all logs
tail -f logs/app_logs.log

# Filter by component
tail -f logs/app_logs.log | grep "RetrieverAgent"
```

---

## 📚 Documentation Links

- **LangGraph**: https://langchain-ai.github.io/langgraph/
- **Pinecone**: https://docs.pinecone.io/
- **Elasticsearch**: https://www.elastic.co/guide/
- **OpenAI**: https://platform.openai.com/docs
- **FastAPI**: https://fastapi.tiangolo.com/

---

## 🤝 Contributing

This is an assignment project. For suggestions or issues, please contact the maintainer.

---

## 📄 License

MIT License - See LICENSE file for details

---

## 🎉 Acknowledgments

Built for the Softvence Agency assignment using:
- Google Gemini for LLM
- LangGraph for orchestration
- Elasticsearch for BM25 search
- Pinecone for semantic search
- FastAPI for REST API

---

## 📞 Support

For issues or questions:
1. Check the documentation guides above
2. Review logs: `logs/app_logs.log`
3. Test individual components
4. Refer to troubleshooting section

---

**Ready to get started?** 🚀

1. Follow [CLOUD_SETUP_GUIDE.md](CLOUD_SETUP_GUIDE.md) to set up cloud services
2. Use [test_cloud_connections.py](test_cloud_connections.py) to verify setup
3. Index your documents with the KB pipeline
4. Start the server and test!
