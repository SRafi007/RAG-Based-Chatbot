# kb_pipeline/data/preprocess.py

import re
from typing import List, Dict
from app.utils.logger import get_logger

logger = get_logger(__name__)


class DocumentPreprocessor:
    """
    Preprocess documents for RAG pipeline.
    - Cleans text
    - Chunks documents
    - Adds metadata
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 128,
        min_chunk_size: int = 100
    ):
        """
        Initialize the preprocessor.

        Args:
            chunk_size: Maximum size of each chunk (in characters)
            chunk_overlap: Overlap between chunks (in characters)
            min_chunk_size: Minimum size for a valid chunk
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        logger.info(
            f"DocumentPreprocessor initialized: chunk_size={chunk_size}, "
            f"overlap={chunk_overlap}"
        )

    def preprocess(self, documents: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Preprocess a list of documents.

        Args:
            documents: List of document dictionaries

        Returns:
            List of preprocessed chunk dictionaries
        """
        all_chunks = []

        for doc in documents:
            # Clean the content
            cleaned_content = self._clean_text(doc['content'])

            # Chunk the document
            chunks = self._chunk_text(cleaned_content)

            # Add metadata to each chunk
            for i, chunk_text in enumerate(chunks):
                if len(chunk_text) >= self.min_chunk_size:
                    chunk = {
                        "content": chunk_text,
                        "source": doc['source'],
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "file_type": doc.get('file_type', 'unknown'),
                        "chunk_id": f"{doc['source']}_chunk_{i}"
                    }
                    all_chunks.append(chunk)

        logger.info(
            f"Preprocessed {len(documents)} documents into {len(all_chunks)} chunks"
        )
        return all_chunks

    def _clean_text(self, text: str) -> str:
        """
        Clean the text by removing extra whitespace, special characters, etc.

        Args:
            text: Raw text

        Returns:
            Cleaned text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,;:?!()\-\'"]+', '', text)

        # Remove multiple consecutive punctuation
        text = re.sub(r'([.,;:?!])\1+', r'\1', text)

        return text.strip()

    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.

        Args:
            text: Text to chunk

        Returns:
            List of text chunks
        """
        chunks = []
        start = 0

        while start < len(text):
            # Calculate end position
            end = start + self.chunk_size

            # Find a good breaking point (end of sentence)
            if end < len(text):
                # Look for sentence boundaries
                boundary = self._find_sentence_boundary(text, end)
                if boundary > start:
                    end = boundary

            # Extract chunk
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # Move to next chunk with overlap
            start = end - self.chunk_overlap
            if start <= 0 or end >= len(text):
                break

        return chunks

    def _find_sentence_boundary(self, text: str, position: int) -> int:
        """
        Find the nearest sentence boundary after a given position.

        Args:
            text: Full text
            position: Target position

        Returns:
            Position of sentence boundary
        """
        # Look ahead for sentence endings
        search_window = text[position:position + 200]

        # Find sentence endings (.!?)
        for match in re.finditer(r'[.!?]\s+', search_window):
            return position + match.end()

        # If no sentence boundary found, return the original position
        return position


if __name__ == "__main__":
    # Test the preprocessor
    test_docs = [
        {
            "content": "This is a test document. It has multiple sentences. "
                       "We want to test the chunking functionality. "
                       "Each chunk should be properly sized. "
                       "And we should handle overlaps correctly. " * 20,
            "source": "test.pdf",
            "file_type": "pdf"
        }
    ]

    preprocessor = DocumentPreprocessor(chunk_size=512, chunk_overlap=128)
    chunks = preprocessor.preprocess(test_docs)

    print(f"Created {len(chunks)} chunks")
    for i, chunk in enumerate(chunks[:3]):  # Print first 3
        print(f"\nChunk {i}:")
        print(f"Source: {chunk['source']}")
        print(f"Content: {chunk['content'][:200]}...")
