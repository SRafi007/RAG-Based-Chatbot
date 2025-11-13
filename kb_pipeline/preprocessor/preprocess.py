# kb_pipeline/preprocessor/preprocess.py

import re
from typing import List, Dict
from app.utils.logger import get_logger

logger = get_logger(__name__)


class DocumentPreprocessor:
    """
    Semantic-aware preprocessor for RAG pipeline.

    Features:
    - Heading-based semantic chunking (## and ### sections)
    - Token-based fallback for large sections
    - Small overlaps (≈50 tokens) for continuity
    - Rich metadata (section, policy_type, source_file, chunk_index, tokens)
    """

    def __init__(
        self,
        target_tokens: int = 350,  # Target ~300-400 tokens per chunk
        max_tokens: int = 450,     # Max tokens before forced split
        overlap_tokens: int = 50,  # Overlap between chunks
        min_tokens: int = 50       # Minimum tokens for a valid chunk
    ):
        """
        Initialize the semantic preprocessor.

        Args:
            target_tokens: Target token count per chunk (300-400 recommended)
            max_tokens: Maximum tokens before forced split
            overlap_tokens: Overlap tokens between chunks (~50)
            min_tokens: Minimum tokens for valid chunk
        """
        self.target_tokens = target_tokens
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.min_tokens = min_tokens

        logger.info(
            f"SemanticPreprocessor initialized: target={target_tokens} tokens, "
            f"overlap={overlap_tokens} tokens"
        )

    def preprocess(self, documents: List[Dict[str, str]]) -> List[Dict[str, any]]:
        """
        Preprocess documents using semantic chunking.

        Args:
            documents: List of document dictionaries with 'content' and 'source'

        Returns:
            List of semantic chunks with rich metadata
        """
        all_chunks = []
        chunk_counter = 0

        for doc in documents:
            content = doc['content']
            source_file = doc['source']

            # Extract policy type from filename
            policy_type = self._extract_policy_type(source_file)

            # Parse into semantic sections
            sections = self._parse_sections(content)

            # Process each section
            for section_data in sections:
                section_chunks = self._chunk_section(
                    section_data,
                    source_file,
                    policy_type,
                    chunk_counter
                )
                all_chunks.extend(section_chunks)
                chunk_counter += len(section_chunks)

        logger.info(
            f"Preprocessed {len(documents)} documents into {len(all_chunks)} semantic chunks"
        )
        return all_chunks

    def _parse_sections(self, content: str) -> List[Dict[str, str]]:
        """
        Parse markdown content into semantic sections based on headings.

        Args:
            content: Full document text

        Returns:
            List of sections with heading and text
        """
        sections = []

        # Split by markdown headings (## or ###)
        heading_pattern = r'^(#{2,3})\s+(.+)$'
        lines = content.split('\n')

        current_heading = None
        current_level = None
        current_text = []

        for line in lines:
            match = re.match(heading_pattern, line)

            if match:
                # Save previous section if exists
                if current_heading and current_text:
                    sections.append({
                        'heading': current_heading,
                        'level': current_level,
                        'text': '\n'.join(current_text).strip()
                    })

                # Start new section
                current_level = len(match.group(1))  # Number of # symbols
                current_heading = match.group(2).strip()
                current_text = [line]  # Include heading in text

            else:
                # Accumulate text for current section
                if current_heading is not None:
                    current_text.append(line)
                else:
                    # Content before first heading (intro/metadata)
                    if line.strip():
                        if current_heading is None:
                            current_heading = "Introduction"
                            current_level = 1
                            current_text = [line]

        # Save last section
        if current_heading and current_text:
            sections.append({
                'heading': current_heading,
                'level': current_level,
                'text': '\n'.join(current_text).strip()
            })

        logger.info(f"Parsed {len(sections)} semantic sections")
        return sections

    def _chunk_section(
        self,
        section: Dict[str, str],
        source_file: str,
        policy_type: str,
        start_index: int
    ) -> List[Dict[str, any]]:
        """
        Chunk a section based on token size with overlap.

        Args:
            section: Section dictionary with heading and text
            source_file: Source filename
            policy_type: Type of policy document
            start_index: Starting chunk index for IDs

        Returns:
            List of chunks with metadata
        """
        heading = section['heading']
        text = section['text']

        # Estimate tokens (rough: 1 token ≈ 4 characters)
        estimated_tokens = self._estimate_tokens(text)

        chunks = []

        # If section is small enough, keep as single chunk
        if estimated_tokens <= self.max_tokens:
            chunk_id = self._generate_chunk_id(source_file, start_index)
            chunks.append({
                "id": chunk_id,
                "text": text,
                "metadata": {
                    "section": heading,
                    "policy_type": policy_type,
                    "source_file": source_file,
                    "chunk_index": start_index,
                    "tokens": estimated_tokens
                }
            })
        else:
            # Split large sections with token-based chunking and overlap
            sub_chunks = self._split_by_tokens(text)

            for i, sub_text in enumerate(sub_chunks):
                chunk_id = self._generate_chunk_id(source_file, start_index + i)
                chunk_tokens = self._estimate_tokens(sub_text)

                chunks.append({
                    "id": chunk_id,
                    "text": sub_text,
                    "metadata": {
                        "section": heading,
                        "policy_type": policy_type,
                        "source_file": source_file,
                        "chunk_index": start_index + i,
                        "tokens": chunk_tokens
                    }
                })

        return chunks

    def _split_by_tokens(self, text: str) -> List[str]:
        """
        Split text by token count with overlap for continuity.

        Args:
            text: Text to split

        Returns:
            List of text chunks with overlap
        """
        # Approximate: 1 token ≈ 4 characters
        char_target = self.target_tokens * 4
        char_overlap = self.overlap_tokens * 4

        chunks = []
        sentences = self._split_sentences(text)

        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            # If adding this sentence exceeds target, save current chunk
            if current_length > 0 and current_length + sentence_length > char_target:
                chunks.append(' '.join(current_chunk))

                # Keep last N characters for overlap
                overlap_text = ' '.join(current_chunk)[-char_overlap:]
                current_chunk = [overlap_text, sentence]
                current_length = len(overlap_text) + sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length

        # Add remaining text
        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.

        Args:
            text: Input text

        Returns:
            List of sentences
        """
        # Split by sentence boundaries (.!?) followed by space or newline
        sentence_pattern = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_pattern, text)
        return [s.strip() for s in sentences if s.strip()]

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count (rough approximation: 1 token ≈ 4 chars).

        Args:
            text: Input text

        Returns:
            Estimated token count
        """
        return len(text) // 4

    def _extract_policy_type(self, filename: str) -> str:
        """
        Extract policy type from filename.

        Args:
            filename: Source filename

        Returns:
            Policy type string
        """
        filename_lower = filename.lower()

        if 'hr' in filename_lower or 'employee' in filename_lower:
            return "HR Policy"
        elif 'security' in filename_lower or 'data' in filename_lower:
            return "Security Policy"
        elif 'handbook' in filename_lower or 'company' in filename_lower:
            return "Company Policy"
        else:
            return "General Policy"

    def _generate_chunk_id(self, source_file: str, index: int) -> str:
        """
        Generate unique chunk ID.

        Args:
            source_file: Source filename
            index: Chunk index

        Returns:
            Unique chunk ID
        """
        # Extract base filename without extension
        base_name = source_file.split('/')[-1].split('.')[0]

        # Create ID: softvence_policy_XXX
        return f"softvence_{base_name}_{index:03d}"


if __name__ == "__main__":
    # Test the semantic preprocessor
    test_doc = """# Softvence Agency - Company Policy Handbook

## 1. Company Overview & Core Values

Softvence Agency is a digital solutions company specializing in web design, AI integration, and software development.

### Core Values

1. Integrity - We act honestly and ethically.
2. Innovation - We encourage new ideas.

---

## 2. Employment & Onboarding

- Eligibility: Applicants must meet local employment laws.
- Probationary Period: 3 months.

### Equal Employment Opportunity

Softvence prohibits discrimination based on race, gender, religion.
"""

    test_docs = [
        {
            "content": test_doc,
            "source": "softvence_company_policy.md"
        }
    ]

    preprocessor = DocumentPreprocessor(
        target_tokens=350,
        overlap_tokens=50
    )
    chunks = preprocessor.preprocess(test_docs)

    print(f"\nCreated {len(chunks)} semantic chunks\n")
    for chunk in chunks[:3]:
        print(f"ID: {chunk['id']}")
        print(f"Section: {chunk['metadata']['section']}")
        print(f"Tokens: {chunk['metadata']['tokens']}")
        print(f"Text preview: {chunk['text'][:150]}...")
        print("-" * 80)
