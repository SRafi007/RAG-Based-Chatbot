# kb_pipeline/__init__.py

"""
Knowledge Base Pipeline for RAG-based Chatbot

This module provides a complete pipeline for:
1. Data Ingestion - Loading documents from various formats
2. Preprocessing - Cleaning and chunking documents
3. Indexing - Creating sparse (BM25) and dense (semantic) indexes
4. Retrieval - Hybrid retrieval combining sparse + dense search
5. Reranking - LLM-based reranking for improved relevance
"""

__version__ = "1.0.0"
