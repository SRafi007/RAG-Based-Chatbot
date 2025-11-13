# kb_pipeline/data/ingest.py

import os
from pathlib import Path
from typing import List, Dict
import PyPDF2
import docx
from app.utils.logger import get_logger

logger = get_logger(__name__)


class DocumentIngester:
    """
    Ingest documents from various formats (PDF, DOCX, TXT) for RAG pipeline.
    """

    def __init__(self, data_dir: str = "data/raw"):
        """
        Initialize the document ingester.

        Args:
            data_dir: Directory containing raw documents
        """
        self.data_dir = Path(data_dir)
        self.supported_formats = ['.pdf', '.docx', '.txt', '.md']
        logger.info(f"DocumentIngester initialized with data_dir: {self.data_dir}")

    def ingest_all(self) -> List[Dict[str, str]]:
        """
        Ingest all supported documents from the data directory.

        Returns:
            List of document dictionaries with metadata
        """
        documents = []

        if not self.data_dir.exists():
            logger.warning(f"Data directory does not exist: {self.data_dir}")
            return documents

        for file_path in self.data_dir.rglob('*'):
            if file_path.suffix.lower() in self.supported_formats:
                try:
                    doc = self.ingest_file(file_path)
                    if doc:
                        documents.append(doc)
                        logger.info(f"Ingested: {file_path.name}")
                except Exception as e:
                    logger.error(f"Failed to ingest {file_path.name}: {e}")

        logger.info(f"Total documents ingested: {len(documents)}")
        return documents

    def ingest_file(self, file_path: Path) -> Dict[str, str]:
        """
        Ingest a single file.

        Args:
            file_path: Path to the file

        Returns:
            Document dictionary with content and metadata
        """
        suffix = file_path.suffix.lower()

        if suffix == '.pdf':
            content = self._read_pdf(file_path)
        elif suffix == '.docx':
            content = self._read_docx(file_path)
        elif suffix in ['.txt', '.md']:
            content = self._read_text(file_path)
        else:
            logger.warning(f"Unsupported file format: {suffix}")
            return None

        if not content or len(content.strip()) == 0:
            logger.warning(f"Empty content in {file_path.name}")
            return None

        return {
            "content": content,
            "source": file_path.name,
            "file_path": str(file_path),
            "file_type": suffix[1:],  # Remove the dot
            "file_size": file_path.stat().st_size
        }

    def _read_pdf(self, file_path: Path) -> str:
        """Read PDF file and extract text."""
        try:
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"Error reading PDF {file_path.name}: {e}")
            return ""

    def _read_docx(self, file_path: Path) -> str:
        """Read DOCX file and extract text."""
        try:
            doc = docx.Document(file_path)
            text = "\n".join([para.text for para in doc.paragraphs])
            return text.strip()
        except Exception as e:
            logger.error(f"Error reading DOCX {file_path.name}: {e}")
            return ""

    def _read_text(self, file_path: Path) -> str:
        """Read plain text or markdown file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read().strip()
        except Exception as e:
            logger.error(f"Error reading text file {file_path.name}: {e}")
            return ""


if __name__ == "__main__":
    # Test the ingester
    ingester = DocumentIngester("data/raw")
    docs = ingester.ingest_all()
    print(f"Ingested {len(docs)} documents")
    for doc in docs[:3]:  # Print first 3
        print(f"\nSource: {doc['source']}")
        print(f"Content preview: {doc['content'][:200]}...")
