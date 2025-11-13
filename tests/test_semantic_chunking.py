"""
Test semantic chunking with the actual Softvence policy document
"""

import re
from typing import List, Dict


class SemanticChunker:
    """Simplified semantic chunker for testing"""

    def __init__(self, target_tokens=350, max_tokens=450, overlap_tokens=50):
        self.target_tokens = target_tokens
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def parse_sections(self, content: str) -> List[Dict[str, str]]:
        """Parse markdown into sections by headings"""
        sections = []
        heading_pattern = r'^(#{2,3})\s+(.+)$'
        lines = content.split('\n')

        current_heading = None
        current_level = None
        current_text = []

        for line in lines:
            match = re.match(heading_pattern, line)

            if match:
                # Save previous section
                if current_heading and current_text:
                    sections.append({
                        'heading': current_heading,
                        'level': current_level,
                        'text': '\n'.join(current_text).strip()
                    })

                # Start new section
                current_level = len(match.group(1))
                current_heading = match.group(2).strip()
                current_text = [line]
            else:
                if current_heading is not None:
                    current_text.append(line)
                elif line.strip():
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

        return sections

    def estimate_tokens(self, text: str) -> int:
        """Estimate tokens (1 token ≈ 4 chars)"""
        return len(text) // 4

    def process_document(self, content: str, source_file: str):
        """Process document and return chunks"""
        sections = self.parse_sections(content)

        print(f"\n{'='*80}")
        print(f"SEMANTIC CHUNKING TEST - {source_file}")
        print(f"{'='*80}\n")
        print(f"Total sections parsed: {len(sections)}\n")

        chunks = []
        for idx, section in enumerate(sections):
            tokens = self.estimate_tokens(section['text'])

            chunk_id = f"softvence_policy_{idx:03d}"
            chunk = {
                "id": chunk_id,
                "text": section['text'],
                "metadata": {
                    "section": section['heading'],
                    "policy_type": "Company Policy",
                    "source_file": source_file,
                    "chunk_index": idx,
                    "tokens": tokens
                }
            }
            chunks.append(chunk)

            # Print section info
            print(f"[Chunk {idx}]")
            print(f"  ID: {chunk_id}")
            print(f"  Section: {section['heading']}")
            print(f"  Level: {'##' if section['level'] == 2 else '###'}")
            print(f"  Tokens: {tokens}")
            print(f"  Text length: {len(section['text'])} chars")
            print(f"  Text preview: {section['text'][:100]}...")
            print()

        return chunks


if __name__ == "__main__":
    # Read the sample policy document
    with open('kb_pipeline/data/raw/sample_policy_handbook.md', 'r', encoding='utf-8') as f:
        content = f.read()

    # Process with semantic chunker
    chunker = SemanticChunker(target_tokens=350, overlap_tokens=50)
    chunks = chunker.process_document(content, "sample_policy_handbook.md")

    # Summary
    print(f"{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"Total chunks created: {len(chunks)}")
    print(f"Average tokens per chunk: {sum(c['metadata']['tokens'] for c in chunks) / len(chunks):.1f}")
    print(f"Min tokens: {min(c['metadata']['tokens'] for c in chunks)}")
    print(f"Max tokens: {max(c['metadata']['tokens'] for c in chunks)}")

    # Show a sample chunk in full
    print(f"\n{'='*80}")
    print(f"SAMPLE CHUNK (full)")
    print(f"{'='*80}")
    sample = chunks[3]  # Show chunk about "Working Hours"
    print(f"ID: {sample['id']}")
    print(f"Section: {sample['metadata']['section']}")
    print(f"Tokens: {sample['metadata']['tokens']}")
    print(f"\nFull text:")
    print(sample['text'])
