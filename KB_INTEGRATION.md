# Knowledge Base Pipeline Integration Guide

## Overview

The RAG-based Chatbot now includes a **Hybrid RAG Pipeline** that combines:
- **Sparse Retrieval**: Elasticsearch with BM25 (keyword-based)
- **Dense Retrieval**: Pinecone with OpenAI embeddings (semantic search)
- **Reranking**: LLM-based relevance scoring

This guide shows how to set up and use the KB pipeline with your chatbot.

---

## Quick Start

### 1. Configure Environment Variables

Update your `.env` file with the required credentials:

```bash
# Elasticsearch (Sparse Vector Retrieval)
ELASTIC_URL=https://your-elastic-cloud-url.es.io:9243
ELASTIC_INDEX=company_policies

# Pinecone (Dense Vector Retrieval)
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_ENV=your-pinecone-environment
PINECONE_INDEX=company-policies

# OpenAI (Embeddings)
OPENAI_API_KEY=your-openai-api-key
EMBEDDING_MODEL=text-embedding-3-small
```

**Getting API Keys:**
- **Elasticsearch**: Sign up at [Elastic Cloud](https://cloud.elastic.co/)
- **Pinecone**: Sign up at [Pinecone](https://www.pinecone.io/)
- **OpenAI**: Get API key from [OpenAI Platform](https://platform.openai.com/api-keys)

### 2. Install Dependencies

Make sure all KB pipeline dependencies are installed:

```bash
pip install elasticsearch pinecone-client openai PyPDF2 python-docx
```

### 3. Add Documents

Place your company policy documents in the `data/raw/` directory:

```bash
mkdir -p data/raw
# Copy your PDF, DOCX, TXT, MD files here
```

Supported formats:
- PDF (`.pdf`)
- Word Documents (`.docx`)
- Text files (`.txt`)
- Markdown files (`.md`)

### 4. Build the Index

Run the indexing pipeline to process and index all documents:

```bash
python -m kb_pipeline.pipeline --mode index --data_dir data/raw
```

This will:
1. Extract text from all documents
2. Clean and chunk the text (512 chars, 128 overlap)
3. Index into Elasticsearch (BM25)
4. Generate embeddings and index into Pinecone

**Expected Output:**
```
[INFO] Starting Knowledge Base Pipeline (Index Mode)
[INFO] Found 15 documents to process
[INFO] Ingested 15 documents
[INFO] Created 347 chunks
[INFO] Indexed 347 chunks to Elasticsearch
[INFO] Indexed 347 chunks to Pinecone
[INFO] Indexing complete! Total chunks: 347
```

### 5. Enable Hybrid Retrieval in RetrieverAgent

The RetrieverAgent starts in **placeholder mode** by default. After indexing your documents, enable hybrid mode:

**Option A: Enable at startup (in `app/orchestrator/agents/retriever_agent.py`)**

```python
# Change this line:
retriever_agent = RetrieverAgent(use_hybrid=False)

# To:
retriever_agent = RetrieverAgent(use_hybrid=True)
```

**Option B: Enable programmatically**

```python
from app.orchestrator.agents.retriever_agent import retriever_agent

# Enable hybrid mode
retriever_agent.enable_hybrid_mode(
    sparse_weight=0.5,  # BM25 weight
    dense_weight=0.5    # Semantic weight
)
```

### 6. Restart the Server

```bash
uvicorn app.main:app --reload
```

---

## Usage Examples

### CLI Search

Test the KB pipeline directly from command line:

```bash
# Search for documents
python -m kb_pipeline.pipeline --mode search --query "What is the remote work policy?" --top_k 5

# Without reranking (faster)
python -m kb_pipeline.pipeline --mode search --query "vacation policy" --top_k 5 --no_rerank
```

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

### API Usage

Once integrated, the chatbot API automatically uses hybrid retrieval:

```bash
curl -X POST http://localhost:8000/api/chat/ \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is the remote work policy?",
    "user_id": "user123"
  }'
```

---

## Configuration

### Retrieval Weights

Adjust the balance between sparse and dense retrieval based on your use case:

```python
# More keyword-focused (better for exact matches)
retriever_agent.enable_hybrid_mode(sparse_weight=0.7, dense_weight=0.3)

# More semantic-focused (better for conceptual queries)
retriever_agent.enable_hybrid_mode(sparse_weight=0.3, dense_weight=0.7)

# Balanced (default)
retriever_agent.enable_hybrid_mode(sparse_weight=0.5, dense_weight=0.5)
```

### Chunk Size

Edit `kb_pipeline/data/preprocess.py` to adjust chunking:

```python
# Smaller chunks = better precision, more chunks
preprocessor = DocumentPreprocessor(chunk_size=256, chunk_overlap=64)

# Larger chunks = more context, fewer chunks
preprocessor = DocumentPreprocessor(chunk_size=1024, chunk_overlap=256)

# Default (balanced)
preprocessor = DocumentPreprocessor(chunk_size=512, chunk_overlap=128)
```

### Reranking

Control LLM-based reranking:

```python
from kb_pipeline.retrieval import Reranker

# LLM reranking (higher quality, slower, more expensive)
reranker = Reranker(use_llm=True)

# Heuristic reranking (faster, cheaper, lower quality)
reranker = Reranker(use_llm=False)
```

---

## Architecture

```
User Query
    |
    v
[DomainGuard] --> Classifies as "policy-related" or "off-topic"
    |
    v (if policy-related)
[RetrieverAgent with HybridRetriever]
    |
    +---> [Elasticsearch BM25] ---+
    |                              |
    +---> [Pinecone Semantic] -----+--> [Weighted Fusion] --> [LLM Reranker]
                                                                      |
                                                                      v
                                                              Top-K Documents
                                                                      |
                                                                      v
                                                              [SummarizerAgent]
                                                                      |
                                                                      v
                                                                Final Response
```

---

## Monitoring

### Check Logs

Monitor pipeline activity:

```bash
# Watch all logs
tail -f logs/app_logs.log

# Filter KB pipeline logs
tail -f logs/app_logs.log | grep "KB Pipeline"

# Filter RetrieverAgent logs
tail -f logs/app_logs.log | grep "RetrieverAgent"
```

### Verify Indexes

**Elasticsearch:**
```python
from kb_pipeline.indexing.index_sparse import SparseIndexer

indexer = SparseIndexer()
print(indexer.client.cat.indices(index=indexer.index_name))
```

**Pinecone:**
```python
from kb_pipeline.indexing.index_dense import DenseIndexer

indexer = DenseIndexer()
stats = indexer.index.describe_index_stats()
print(f"Total vectors: {stats['total_vector_count']}")
```

---

## Troubleshooting

### Issue: "RetrieverAgent initialized in placeholder mode"

**Cause**: Hybrid mode is disabled or failed to initialize.

**Solution**:
1. Check that environment variables are set in `.env`
2. Verify API credentials are valid
3. Ensure documents are indexed: `python -m kb_pipeline.pipeline --mode index --data_dir data/raw`
4. Enable hybrid mode: `retriever_agent.enable_hybrid_mode()`

### Issue: Elasticsearch connection error

**Check:**
```python
from kb_pipeline.indexing.index_sparse import SparseIndexer
indexer = SparseIndexer()
# Should not raise an error
```

**Common fixes:**
- Verify `ELASTIC_URL` in `.env` is correct
- Check network connectivity to Elasticsearch
- Verify API key/credentials

### Issue: Pinecone index not found

**Check:**
```python
from pinecone import Pinecone
pc = Pinecone(api_key="your-key")
print(pc.list_indexes())
```

**Solution:**
Create index manually:
```python
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key="your-key")
pc.create_index(
    name="company-policies",
    dimension=1536,  # text-embedding-3-small dimension
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1")
)
```

### Issue: No documents retrieved

**Check:**
1. Verify documents are in `data/raw/`
2. Run indexing: `python -m kb_pipeline.pipeline --mode index --data_dir data/raw`
3. Check index stats (see Monitoring section)

---

## Performance Tips

1. **Optimize Chunk Size**: Experiment with different sizes based on your document structure
2. **Adjust Weights**: Tune sparse/dense weights for your specific use case
3. **Batch Indexing**: Index in batches if you have many documents
4. **Cache Results**: Consider caching frequent queries
5. **Monitor Costs**: OpenAI embeddings and LLM reranking incur API costs

---

## Next Steps

1. **Add More Documents**: Continuously update your knowledge base
2. **Monitor Performance**: Track retrieval quality and adjust weights
3. **Implement Feedback Loop**: Collect user feedback to improve retrieval
4. **Scale Infrastructure**: Use production-grade Elasticsearch and Pinecone clusters
5. **Add Document Versioning**: Track document updates over time

---

## Support

For issues or questions:
- Check [kb_pipeline/README.md](kb_pipeline/README.md) for detailed pipeline documentation
- Review [ARCHITECTURE.md](ARCHITECTURE.md) for system design
- Check logs: `logs/app_logs.log`

---

## Summary

The hybrid RAG pipeline enhances your chatbot with:
- **Better Retrieval**: Combines keyword and semantic search
- **Higher Quality**: LLM-based reranking improves relevance
- **Scalability**: Production-ready infrastructure (Elasticsearch + Pinecone)
- **Flexibility**: Easy to configure and extend

Start by indexing your documents, enable hybrid mode, and your chatbot will automatically use the improved retrieval system!
