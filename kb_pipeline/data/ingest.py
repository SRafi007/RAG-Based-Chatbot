# kb_pipeline/data/ingest.py

"""
Document ingestion module for loading raw files.
"""

from pathlib import Path
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentIngester:
    """
    Ingests documents from various file formats.
    """

    def __init__(self, data_dir: str = "kb_pipeline/data/raw"):
        """
        Initialize ingester.

        Args:
            data_dir: Directory containing raw documents
        """
        self.data_dir = Path(data_dir)
        logger.info(f"Initializing DocumentIngester with data_dir: {self.data_dir}")

    def ingest_markdown(self, file_path: Path) -> Dict[str, str]:
        """
        Ingest a markdown file.

        Args:
            file_path: Path to markdown file

        Returns:
            Document dictionary
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            return {
                "id": file_path.stem,
                "content": content,
                "source_file": file_path.name,
                "file_type": "markdown"
            }
        except Exception as e:
            logger.error(f"Error ingesting {file_path}: {e}")
            return None

    def ingest_all(self) -> List[Dict[str, str]]:
        """
        Ingest all documents from the data directory.

        Returns:
            List of document dictionaries
        """
        documents = []

        if not self.data_dir.exists():
            logger.error(f"Data directory does not exist: {self.data_dir}")
            return documents

        logger.info(f"Scanning for documents in: {self.data_dir}")

        # Find all markdown files
        md_files = list(self.data_dir.glob("*.md"))
        logger.info(f"Found {len(md_files)} markdown files")

        for file_path in md_files:
            logger.info(f"Ingesting: {file_path.name}")
            doc = self.ingest_markdown(file_path)
            if doc:
                documents.append(doc)

        logger.info(f"Successfully ingested {len(documents)} documents")
        return documents


if __name__ == "__main__":
    # Test ingester
    ingester = DocumentIngester()
    docs = ingester.ingest_all()
    print(f"\nIngested {len(docs)} documents:")
    for doc in docs:
        print(f"  - {doc['source_file']} ({len(doc['content'])} characters)")
