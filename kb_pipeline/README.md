# Knowledge Base Pipeline

Hybrid RAG pipeline combining sparse (BM25) and dense (semantic) retrieval for company policy documents.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    KB PIPELINE FLOW                         │
└─────────────────────────────────────────────────────────────┘

1. INGESTION
   ├── PDF, DOCX, TXT, MD files
   └── Extract text content

2. PREPROCESSING
   ├── Clean text
   ├── Chunk documents (512 chars, 128 overlap)
   └── Add metadata

3. INDEXING
   ├── Sparse (Elasticsearch + BM25)
   │   └── Keyword-based search
   └── Dense (Pinecone + OpenAI Embeddings)
       └── Semantic search

4. RETRIEVAL
   ├── Hybrid Retriever
   │   ├── Sparse results (BM25)
   │   ├── Dense results (Embeddings)
   │   └── Weighted fusion
   └── Reranker (LLM-based)

5. RESPONSE
   └── Top-K most relevant chunks
```

## Directory Structure

```
kb_pipeline/
├── data/
│   ├── ingest.py           # Document ingestion
│   └── preprocess.py       # Text cleaning & chunking
├── indexing/
│   ├── index_sparse.py     # Elasticsearch (BM25)
│   └── index_dense.py      # Pinecone (Embeddings)
├── retrieval/
│   ├── hybrid_retriever.py # Combine sparse + dense
│   ├── reranker.py         # LLM reranking
│   └── __init__.py
├── pipeline.py             # Main orchestrator
└── README.md               # This file
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Update `.env` with your credentials:

```bash
# Elasticsearch (Sparse Vector)
ELASTIC_URL=https://your-elastic-cloud-url.es.io:9243
ELASTIC_INDEX=company_policies

# Pinecone (Dense Vector)
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_ENV=your-pinecone-environment
PINECONE_INDEX=company-policies

# OpenAI (Embeddings)
OPENAI_API_KEY=your-openai-api-key
EMBEDDING_MODEL=text-embedding-3-small
```

### 3. Prepare Documents

Place your company policy documents in `data/raw/`:

```bash
mkdir -p data/raw
# Copy your PDF, DOCX, TXT, MD files here
```

## Usage

### Build Index

Index all documents in the knowledge base:

```bash
python -m kb_pipeline.pipeline --mode index --data_dir data/raw
```

This will:
1. Ingest all documents from `data/raw/`
2. Preprocess and chunk them
3. Index into Elasticsearch (sparse)
4. Index into Pinecone (dense)

### Search

Query the knowledge base:

```bash
python -m kb_pipeline.pipeline --mode search --query "What is the remote work policy?" --top_k 5
```

Options:
- `--query`: Your search query (required)
- `--top_k`: Number of results (default: 5)
- `--no_rerank`: Disable LLM reranking

### Programmatic Usage

```python
from kb_pipeline.pipeline import KnowledgeBasePipeline

# Initialize pipeline
pipeline = KnowledgeBasePipeline()

# Build index
pipeline.build_index("data/raw")

# Search
results = pipeline.search(
    query="What is the vacation policy?",
    top_k=5,
    use_reranking=True
)

# Display results
print(pipeline.format_results(results))
```

## Components

### 1. DocumentIngester

Reads documents from multiple formats:
- PDF (using PyPDF2)
- DOCX (using python-docx)
- TXT, MD (plain text)

### 2. DocumentPreprocessor

Prepares documents for indexing:
- Cleans text (removes extra whitespace, special chars)
- Chunks into overlapping segments
- Adds metadata (source, chunk_id, etc.)

### 3. SparseIndexer (Elasticsearch)

Traditional keyword search:
- BM25 algorithm
- Fast exact matching
- Good for specific terms

### 4. DenseIndexer (Pinecone)

Semantic search:
- OpenAI embeddings (text-embedding-3-small)
- Cosine similarity
- Captures meaning and context

### 5. HybridRetriever

Combines both methods:
- Weighted fusion (default: 50% sparse, 50% dense)
- Reciprocal rank fusion
- Deduplication

### 6. Reranker

Second-stage ranking:
- LLM-based relevance scoring
- Improves precision
- Optional heuristic fallback

## Integration with Orchestrator

Update `RetrieverAgent` to use the hybrid retriever:

```python
# app/orchestrator/agents/retriever_agent.py

from kb_pipeline.retrieval import HybridRetriever, Reranker

class RetrieverAgent:
    def __init__(self):
        self.retriever = HybridRetriever()
        self.reranker = Reranker(use_llm=True)

    def __call__(self, state: AgentState) -> AgentState:
        query = state["message"]

        # Hybrid retrieval
        results = self.retriever.retrieve(query, top_k=10)

        # Rerank
        results = self.reranker.rerank(query, results, top_k=3)

        # Format context
        context = self._format_context(results)

        state["retrieved_docs"] = results
        state["context"] = context
        return state
```

## Performance Tips

### 1. Chunk Size

- **Smaller chunks** (256-512): Better precision, more chunks
- **Larger chunks** (1024+): More context, fewer chunks

### 2. Retrieval Weights

Adjust based on your use case:
```python
# More keyword-focused
retriever = HybridRetriever(sparse_weight=0.7, dense_weight=0.3)

# More semantic-focused
retriever = HybridRetriever(sparse_weight=0.3, dense_weight=0.7)
```

### 3. Reranking

- **LLM reranking**: Higher quality, slower, more expensive
- **Heuristic reranking**: Faster, cheaper, lower quality

## Monitoring

Check logs for pipeline status:

```bash
tail -f logs/app_logs.log | grep "KB Pipeline"
```

## Troubleshooting

### Elasticsearch Connection Error

```python
# Check connection
from kb_pipeline.indexing.index_sparse import SparseIndexer
indexer = SparseIndexer()
```

### Pinecone Index Not Found

```python
# List indexes
from pinecone import Pinecone
pc = Pinecone(api_key="your-key")
print(pc.list_indexes())
```

### No Documents Found

```bash
# Verify data directory
ls -la data/raw/
```

## License

MIT
