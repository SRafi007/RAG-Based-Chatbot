# RAG-based Chatbot Architecture

## 📋 Overview

This is a RAG (Retrieval-Augmented Generation) based chatbot built with LangGraph for company policy questions. The system uses a multi-agent architecture with plug-and-play capabilities.

## 🏗️ Architecture

### High-Level Workflow

```
User Query
    ↓
DomainGuard (Policy Scope Checker)
    ↓
Router (LangGraph Conditional Edge)
    ↓
    ├─→ "off-topic" → SummarizerAgent → END
    │                      ↓
    │                 Soft Refusal + Warning
    │
    └─→ "policy-related" → RetrieverAgent → SummarizerAgent → END
                              ↓                    ↓
                         Retrieve Docs        Generate Answer
```

### Directory Structure

```
app/
├── orchestrator/              # LangGraph-based orchestration
│   ├── __init__.py
│   ├── state.py              # AgentState TypedDict
│   ├── orchestrator.py       # LangGraph workflow
│   └── agents/               # Plug-and-play agents
│       ├── __init__.py
│       ├── domain_guard.py   # Policy scope checker
│       ├── retriever_agent.py # RAG document retriever
│       └── summarizer_agent.py # Response generator
│
├── api/
│   └── chat_api.py           # FastAPI endpoints
│
├── models/
│   ├── __init__.py
│   └── conversation.py       # SQLAlchemy models (PostgreSQL)
│
├── schemas/
│   ├── __init__.py
│   └── conversation.py       # Pydantic schemas
│
├── memory/
│   ├── __init__.py
│   └── short_term_memory.py  # Redis-based session memory
│
├── config/
│   ├── db.py                 # PostgreSQL connection
│   └── settings.py           # Environment variables
│
├── utils/
│   ├── __init__.py
│   ├── llm_client.py         # Gemini API client
│   ├── logger.py             # Logging setup
│   └── redis_client.py       # Redis connection
│
└── main.py                   # FastAPI application
```

## 🔧 Components

### 1. Orchestrator (`app/orchestrator/orchestrator.py`)

**LangGraph-based workflow manager**

- Built with `langgraph.graph.StateGraph`
- Manages agent execution flow
- Provides both standard and streaming processing
- Easy to extend with new agents

**Key Methods:**
- `process(user_id, message, session_id)` - Standard synchronous processing
- `stream_process(user_id, message, session_id)` - Streaming response processing

### 2. Agents (`app/orchestrator/agents/`)

#### DomainGuard Agent
- **Purpose**: Classify queries as "policy-related" or "off-topic"
- **LLM**: Uses Gemini for intelligent classification
- **Output**: Updates state with classification

#### Retriever Agent
- **Purpose**: Retrieve relevant policy documents
- **Technology**: Vector search (ChromaDB/FAISS - to be integrated)
- **Output**: Retrieved documents and formatted context

#### Summarizer Agent
- **Purpose**: Generate final responses
- **Modes**:
  - Policy-related: Uses retrieved context to answer
  - Off-topic: Returns soft refusal with helpful warning
- **Streaming**: Supports async streaming responses

### 3. State Management

#### Short-Term Memory (Redis)
- Session-based conversation history
- Fast access for real-time chat
- TTL-based expiration (configurable)
- Managed by `app/memory/short_term_memory.py`

#### Long-Term Storage (PostgreSQL)
- Persistent conversation history
- Analytics and reporting
- Two tables:
  - `conversations` - Individual messages
  - `conversation_sessions` - Session metadata

### 4. API Endpoints

**POST /api/chat/**
- Standard chat endpoint
- Returns complete response
- Saves to both Redis and PostgreSQL

**POST /api/chat/stream**
- Streaming chat endpoint
- SSE-compatible
- Real-time token-by-token response

**GET /api/chat/history/{session_id}**
- Retrieve conversation history
- Fetches from PostgreSQL

## 🔌 Plug-and-Play Agent System

### Adding a New Agent

1. **Create Agent File**
```python
# app/orchestrator/agents/my_new_agent.py

from app.orchestrator.state import AgentState
from app.utils.logger import get_logger

logger = get_logger(__name__)

class MyNewAgent:
    def __call__(self, state: AgentState) -> AgentState:
        # Your agent logic here
        logger.info("MyNewAgent: Processing...")

        # Update state
        state["some_field"] = "some_value"

        return state

# Singleton
my_new_agent = MyNewAgent()
```

2. **Update State Schema** (if needed)
```python
# app/orchestrator/state.py

class AgentState(TypedDict):
    # ... existing fields ...
    some_field: Optional[str]  # Add new field
```

3. **Register in Orchestrator**
```python
# app/orchestrator/orchestrator.py

from app.orchestrator.agents import my_new_agent

def _build_graph(self):
    workflow = StateGraph(AgentState)

    # Add your new node
    workflow.add_node("my_new_agent", my_new_agent)

    # Add edges
    workflow.add_edge("some_node", "my_new_agent")
    workflow.add_edge("my_new_agent", "next_node")
```

## 🗄️ Database Models

### Conversation
```python
{
    "id": int,
    "session_id": str,
    "user_id": str,
    "role": str,  # "user" or "assistant"
    "message": str,
    "classification": str,  # "policy-related", "off-topic"
    "metadata": dict,  # JSON field
    "created_at": datetime
}
```

### ConversationSession
```python
{
    "id": int,
    "session_id": str,
    "user_id": str,
    "message_count": int,
    "is_active": bool,
    "started_at": datetime,
    "last_activity_at": datetime,
    "ended_at": datetime
}
```

## 🧪 Testing

### Test Orchestrator
```bash
python -m tests.test_new_orchestrator
```

### Test LLM Client
```bash
python -m tests.test_llm
```

### Test Redis STM
```bash
python -m tests.redis_stm
```

## 🚀 Running the Application

### Development
```bash
uvicorn app.main:app --reload --port 8000
```

### Production
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## 📝 Environment Variables

Required in `.env`:
```bash
# App
APP_NAME=RAG-based Chatbot
APP_ENV=development
APP_PORT=8000

# PostgreSQL
DATABASE_URL=postgresql://user:pass@host:port/db

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_USERNAME=default
REDIS_PASSWORD=your_password
REDIS_DB=0
REDIS_USE_TLS=false

# Gemini API
GEMINI_API_KEY=your_api_key
GEMINI_MODEL_NAME=gemini-2.5-flash

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/app_logs.log

# Memory Config
MAX_SESSION_MESSAGES=200
SESSION_TTL_DAYS=30
```

## 🔮 Future Enhancements

- [ ] Integrate ChromaDB/FAISS for actual vector search
- [ ] Add document ingestion pipeline
- [ ] Implement feedback mechanism
- [ ] Add analytics dashboard
- [ ] Support for document upload
- [ ] Multi-language support
- [ ] Add more specialized agents (e.g., AnalyticsAgent, FeedbackAgent)

## 📚 Tech Stack

- **Framework**: FastAPI
- **Orchestration**: LangGraph
- **LLM**: Google Gemini
- **Vector DB**: ChromaDB/FAISS (planned)
- **Cache**: Redis
- **Database**: PostgreSQL
- **ORM**: SQLAlchemy
- **Validation**: Pydantic
