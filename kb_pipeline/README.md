# Knowledge Base Pipeline

Hybrid RAG system combining sparse (BM25) and dense (semantic) retrieval for company policy documents.

## System Architecture

```
Data Ingestion → Semantic Chunking → Dual Indexing → Hybrid Retrieval → LLM Response
     |                  |                   |                 |
  MD files      350 tokens/chunk    Elasticsearch      Weighted Fusion
                50-token overlap       Pinecone         + Reranking
```

## Features

- **Semantic Chunking**: Heading-based splitting with token fallback
- **Hybrid Search**: BM25 (keyword) + all-mpnet-base-v2 (semantic)
- **Free Embeddings**: Local sentence-transformers (768D, no API limits)
- **Memory System**: Redis (STM) + PostgreSQL (LTM)
- **LLM**: Gemini 2.5 Flash with tool use

## Directory Structure

```
kb_pipeline/
├── data/
│   ├── ingest.py              # Load MD/PDF/DOCX files
│   └── raw/                   # Place documents here
├── preprocessor/
│   └── preprocess.py          # Semantic chunking
├── indexing/
│   ├── index_sparse.py        # Elasticsearch (BM25)
│   └── index_dense.py         # Pinecone (embeddings)
├── retrieval/
│   ├── hybrid_retriever.py    # Combine sparse + dense
│   └── reranker.py            # LLM-based reranking
└── pipeline.py                # Main orchestrator
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Update `.env`:

```bash

# Pinecone
PINECONE_API_KEY=your-api-key
PINECONE_HOST=https://your-index.pinecone.io
PINECONE_INDEX=company-policies

# Local Embeddings (FREE)
EMBEDDING_MODEL=all-mpnet-base-v2
EMBEDDING_DIMENSION=768

# Gemini LLM
GEMINI_API_KEY=your-api-key
GEMINI_MODEL_NAME=gemini-2.5-flash

# Redis
REDIS_HOST=your-redis-host
REDIS_PORT=17369
REDIS_PASSWORD=your-password

# PostgreSQL
DATABASE_URL=postgresql://user:pass@host:port/db
```

### 3. Index Documents

Place documents in `kb_pipeline/data/raw/` and run:

```bash
python test_index_documents.py
```

Or use the pipeline:

```bash
python -m kb_pipeline.pipeline --mode index --data_dir kb_pipeline/data/raw
```

### 4. Search

```bash
python -m kb_pipeline.pipeline --mode search --query "What is the remote work policy?" --top_k 5
```

Options:
- `--query`: Search query (required)
- `--top_k`: Number of results (default: 5)
- `--no_rerank`: Disable LLM reranking

### 5. Start API Server

```bash
python -m app.main
```

Access the API at `http://localhost:8000/docs`

## Components

### Semantic Chunking

**DocumentPreprocessor** splits documents intelligently:
- Primary: Markdown heading boundaries (##, ###)
- Fallback: Token-based splitting (~350 tokens)
- Overlap: 50 tokens for continuity
- Metadata: section name, policy type, source file, token count

### Sparse Indexing (Elasticsearch)

**SparseIndexer** provides keyword search:
- Algorithm: BM25
- Best for: Exact terms, acronyms, names
- Index: `company_policies`

### Dense Indexing (Pinecone)

**DenseIndexer** provides semantic search:
- Model: all-mpnet-base-v2 (768D)
- Free: No API limits, runs locally
- Best for: Meaning, context, paraphrases
- Metric: Cosine similarity

### Hybrid Retrieval

**HybridRetriever** combines both:
- Default weights: 50% sparse, 50% dense
- Fusion: Reciprocal Rank Fusion (RRF)
- Deduplication: Removes redundant chunks

### Reranking

**Reranker** improves result quality:
- Method: LLM-based relevance scoring
- Model: Gemini 2.5 Flash
- Fallback: Heuristic scoring

## Programmatic Usage

```python
from kb_pipeline.pipeline import KnowledgeBasePipeline

# Initialize
pipeline = KnowledgeBasePipeline()

# Build index
indexed = pipeline.build_index("kb_pipeline/data/raw")
print(f"Indexed {indexed} chunks")

# Search
results = pipeline.search(
    query="What is the vacation policy?",
    top_k=5,
    use_reranking=True
)

# Display
print(pipeline.format_results(results))
```

## Orchestrator Integration

Update `app/orchestrator/agents/retriever_agent.py`:

```python
from kb_pipeline.retrieval.hybrid_retriever import HybridRetriever
from kb_pipeline.retrieval.reranker import Reranker

class RetrieverAgent:
    def __init__(self, use_hybrid=True):
        self.retriever = HybridRetriever(
            sparse_weight=0.5,
            dense_weight=0.5
        )
        self.reranker = Reranker(use_llm=True)

    def __call__(self, state: AgentState) -> AgentState:
        query = state["message"]

        # Hybrid retrieval
        results = self.retriever.retrieve(query, top_k=10)

        # Rerank
        results = self.reranker.rerank(query, results, top_k=3)

        state["retrieved_docs"] = results
        state["context"] = self._format_context(results)
        return state
```

## Configuration Tuning

### Chunk Size

Edit `kb_pipeline/pipeline.py`:

```python
self.preprocessor = DocumentPreprocessor(
    target_tokens=350,  # Increase for more context
    max_tokens=450,     # Upper limit
    overlap_tokens=50,  # Continuity
    min_tokens=50       # Lower limit
)
```

### Retrieval Weights

Adjust for your use case:

```python
# Keyword-focused (technical terms, names)
retriever = HybridRetriever(sparse_weight=0.7, dense_weight=0.3)

# Semantic-focused (concepts, paraphrases)
retriever = HybridRetriever(sparse_weight=0.3, dense_weight=0.7)

# Balanced (default)
retriever = HybridRetriever(sparse_weight=0.5, dense_weight=0.5)
```

## System Status Check

```bash
# Check Elasticsearch
curl -u elastic:$ELASTIC_API_KEY $ELASTIC_URL

# Check Pinecone
python -c "from pinecone import Pinecone; pc = Pinecone(api_key='$PINECONE_API_KEY'); print(pc.list_indexes())"

# Check Redis
redis-cli -h $REDIS_HOST -p $REDIS_PORT -a $REDIS_PASSWORD PING

# Check PostgreSQL
psql $DATABASE_URL -c "SELECT 1"

# View logs
tail -f logs/app_logs.log
```

## Troubleshooting

### Connection Errors

**Elasticsearch:**
```python
from kb_pipeline.indexing.index_sparse import SparseIndexer
indexer = SparseIndexer()  # Check connection logs
```

**Pinecone:**
```python
from kb_pipeline.indexing.index_dense import DenseIndexer
indexer = DenseIndexer()  # Check connection logs
```

### No Documents Found

```bash
ls -la kb_pipeline/data/raw/
# Ensure .md files are present
```

### Embedding Model Download

First run downloads all-mpnet-base-v2 (~400MB). Subsequent runs use cached model.

### Circular Import Errors

The pipeline components use standard `logging` instead of `app.utils.logger` to avoid circular dependencies.

## Performance

- **Indexing Speed**: ~2-3 chunks/second (local embeddings)
- **Search Latency**:
  - Sparse only: ~50ms
  - Dense only: ~100ms
  - Hybrid + rerank: ~500ms
- **Memory Usage**: ~1GB (model loaded)

## Current Implementation

**Indexed:**
- Documents: 1 (sample_policy_handbook.md)
- Chunks: 27 semantic sections
- Pinecone: 27 vectors (768D)
- Elasticsearch: Ready (configure URL in .env)

**Models:**
- Embeddings: all-mpnet-base-v2 (FREE, unlimited)
- LLM: Gemini 2.5 Flash
- Memory: Redis + PostgreSQL

## Next Steps

1. Fix Elasticsearch URL in `.env` (line 43)
2. Re-index documents: `python test_index_documents.py`
3. Test hybrid search: `python -m kb_pipeline.pipeline --mode search --query "company values"`
4. Start API server: `python -m app.main`
5. Add more documents to `kb_pipeline/data/raw/`

## License

MIT
