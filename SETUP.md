# RAG-based Chatbot - Setup Guide

## 📦 Installation

### 1. Create Virtual Environment

```bash
python -m venv rag_env
```

### 2. Activate Virtual Environment

**Windows:**
```bash
rag_env\Scripts\activate
```

**Linux/Mac:**
```bash
source rag_env/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## ⚙️ Configuration

### 1. Create `.env` File

Copy the example below and update with your credentials:

```bash
# App Configuration
APP_NAME=RAG-based Chatbot
APP_ENV=development
APP_PORT=8000

# PostgreSQL Database (Supabase)
DATABASE_URL=postgresql://user:password@host:port/database

# Redis Configuration
REDIS_HOST=your-redis-host.com
REDIS_PORT=17369
REDIS_USERNAME=default
REDIS_PASSWORD=your_redis_password
REDIS_DB=0
REDIS_USE_TLS=false

# Google Gemini API
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL_NAME=gemini-2.5-flash
GEMINI_THINKING_BUDGET=-1
GEMINI_IMAGE_SIZE=1K

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/app_logs.log

# Memory Configuration
MAX_SESSION_MESSAGES=200
SESSION_TTL_DAYS=30
```

### 2. Create Logs Directory

```bash
mkdir logs
```

## 🗄️ Database Setup

### Initialize PostgreSQL Tables

The application will automatically create tables on startup. The following tables will be created:

- `conversations` - Stores individual chat messages
- `conversation_sessions` - Stores session metadata

**Models are defined in:**
- `app/models/conversation.py`

## 🧪 Testing

### Test LLM Client

```bash
python -m tests.test_llm
```

### Test Orchestrator

```bash
python -m tests.test_new_orchestrator
```

### Test Redis STM

```bash
python -m tests.redis_stm
```

## 🚀 Running the Application

### Development Mode (with auto-reload)

```bash
uvicorn app.main:app --reload --port 8000
```

### Production Mode

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## 📡 API Endpoints

Once running, the API will be available at `http://localhost:8000`

### Chat Endpoints

**Standard Chat:**
```bash
POST http://localhost:8000/api/chat/
Content-Type: application/json

{
  "message": "What is the remote work policy?",
  "user_id": "user123",
  "session_id": "optional-session-id"
}
```

**Streaming Chat:**
```bash
POST http://localhost:8000/api/chat/stream
Content-Type: application/json

{
  "message": "Tell me about vacation policies",
  "user_id": "user123"
}
```

**Get History:**
```bash
GET http://localhost:8000/api/chat/history/{session_id}
```

### Interactive API Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🔧 Project Structure

```
RAG-based Chatbot/
├── app/
│   ├── __init__.py
│   ├── main.py                      # FastAPI application
│   ├── api/                         # API endpoints
│   │   ├── __init__.py
│   │   └── chat_api.py
│   ├── orchestrator/                # LangGraph orchestration
│   │   ├── __init__.py
│   │   ├── state.py                 # State schema
│   │   ├── orchestrator.py          # Workflow
│   │   └── agents/                  # Agents
│   │       ├── __init__.py
│   │       ├── domain_guard.py
│   │       ├── retriever_agent.py
│   │       └── summarizer_agent.py
│   ├── models/                      # SQLAlchemy models
│   │   ├── __init__.py
│   │   └── conversation.py
│   ├── schemas/                     # Pydantic schemas
│   │   ├── __init__.py
│   │   └── conversation.py
│   ├── memory/                      # Redis STM
│   │   ├── __init__.py
│   │   └── short_term_memory.py
│   ├── config/                      # Configuration
│   │   ├── __init__.py
│   │   ├── db.py
│   │   └── settings.py
│   └── utils/                       # Utilities
│       ├── __init__.py
│       ├── llm_client.py
│       ├── logger.py
│       └── redis_client.py
├── tests/                           # Test files
│   ├── test_llm.py
│   ├── test_new_orchestrator.py
│   └── redis_stm.py
├── logs/                            # Log files
├── requirements.txt                 # Dependencies
├── .env                             # Environment variables
├── ARCHITECTURE.md                  # Architecture documentation
└── SETUP.md                         # This file
```

## 🐛 Troubleshooting

### Import Errors

If you encounter import errors, ensure:
1. Virtual environment is activated
2. All dependencies are installed: `pip install -r requirements.txt`
3. You're running from the project root directory

### Database Connection Issues

Check:
1. PostgreSQL DATABASE_URL in `.env` is correct
2. Database server is running and accessible
3. Credentials are valid

### Redis Connection Issues

Check:
1. Redis host and port in `.env` are correct
2. Redis server is running
3. Username/password are valid
4. Network access to Redis server

### Gemini API Issues

Check:
1. GEMINI_API_KEY is set correctly in `.env`
2. API key is active and has quota
3. Network connectivity to Google APIs

## 📚 Next Steps

1. **Add Vector Store**: Integrate ChromaDB or FAISS in `retriever_agent.py`
2. **Upload Documents**: Create document ingestion pipeline
3. **Frontend**: Build a chat UI
4. **Deployment**: Deploy to cloud platform (AWS, GCP, Azure)

## 🆘 Support

For issues or questions:
1. Check the [ARCHITECTURE.md](ARCHITECTURE.md) for system design
2. Review the code comments
3. Check logs in `logs/app_logs.log`
