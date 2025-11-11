#!/usr/bin/env python3
"""
Chunk markdown files from Rag_Vault/articles/DOC_paper_* directories.

Supports three chunking methods:
1. RecursiveCharacterSplitter
2. MarkdownTextSplitter
3. SemanticChunker
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownTextSplitter,
)
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings


class DocumentChunker:
    """Chunk markdown files with metadata from registry CSV."""
    
    REGISTRY_PATH = Path("/home/ubuntu/LIA_CARA/src/LIACARA/Rag_Vault/registry/document_master_list.csv")
    ARTICLES_ROOT = Path("/home/ubuntu/LIA_CARA/src/LIACARA/Rag_Vault/articles")
    # Registry path as it should appear in output (user-specified format)
    # Note: This is the logical path format, not necessarily the filesystem path
    REGISTRY_PATH_OUTPUT = "/LIACARA/Rag_Vault/registry/document_master_list.csv"
    
    def __init__(self, method: str = "recursive", chunk_size: int = 1000, chunk_overlap: int = 200, embeddings=None):
        """
        Initialize chunker with specified method.
        
        Args:
            method: One of 'recursive', 'markdown', or 'semantic'
            chunk_size: Size of chunks (for recursive and markdown methods)
            chunk_overlap: Overlap between chunks (for recursive and markdown methods)
            embeddings: Optional embeddings model for semantic chunking (required if method='semantic')
        """
        self.method = method.lower()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embeddings = embeddings
        self.registry_data = self._load_registry()
        self.splitter = self._create_splitter()
        
    def _load_registry(self) -> Dict[str, Dict[str, Any]]:
        """Load document metadata from registry CSV."""
        registry = {}
        registry_path = DocumentChunker.REGISTRY_PATH
        if not registry_path.exists():
            raise FileNotFoundError(f"Registry not found: {registry_path}")
        
        with open(registry_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                doc_id = row['doc_id']
                # Parse site_ids and concept_ids (pipe-separated)
                site_ids = [s.strip() for s in row['site_ids'].split('|') if s.strip()] if row.get('site_ids') else []
                concept_ids = [c.strip() for c in row['concept_ids'].split('|') if c.strip()] if row.get('concept_ids') else []
                
                registry[doc_id] = {
                    'site_ids': site_ids,
                    'concept_ids': concept_ids,
                    'license': row.get('license', ''),
                    'checksum_sha256': row.get('checksum_sha256', ''),
                }
        return registry
    
    def _create_splitter(self):
        """Create the appropriate text splitter based on method."""
        if self.method == "recursive":
            return RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", " ", ""]
            )
        elif self.method == "markdown":
            return MarkdownTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
        elif self.method == "semantic":
            # SemanticChunker requires embeddings
            if self.embeddings is None:
                # Try to use OpenAI embeddings as default
                try:
                    self.embeddings = OpenAIEmbeddings()
                except Exception as e:
                    raise RuntimeError(
                        f"SemanticChunker requires embeddings. "
                        f"Either provide embeddings parameter or set OPENAI_API_KEY. Error: {e}"
                    )
            return SemanticChunker(
                embeddings=self.embeddings,
                buffer_size=1,  # Number of sentences to combine
            )
        else:
            raise ValueError(f"Unknown method: {self.method}. Choose from: recursive, markdown, semantic")
    
    def _calculate_sha256(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def _extract_doc_id_from_path(self, md_path: Path) -> str:
        """Extract doc_id from markdown file path (e.g., DOC_paper_01.md -> DOC_paper_01)."""
        return md_path.stem
    
    def _get_doc_metadata(self, doc_id: str) -> Dict[str, Any]:
        """Get metadata for a document from registry."""
        if doc_id not in self.registry_data:
            print(f"Warning: doc_id '{doc_id}' not found in registry. Using empty metadata.")
            return {
                'site_ids': [],
                'concept_ids': [],
                'license': '',
                'checksum_sha256': '',
            }
        return self.registry_data[doc_id]
    
    def chunk_file(self, md_path: Path) -> List[Dict[str, Any]]:
        """
        Chunk a single markdown file and return list of chunk dictionaries.
        
        Args:
            md_path: Path to markdown file
            
        Returns:
            List of chunk dictionaries with all required fields
        """
        # Read markdown content
        text = md_path.read_text(encoding='utf-8', errors='ignore')
        
        # Calculate checksum
        checksum = self._calculate_sha256(md_path)
        
        # Extract doc_id
        doc_id = self._extract_doc_id_from_path(md_path)
        
        # Get metadata from registry
        metadata = self._get_doc_metadata(doc_id)
        
        # Use registry checksum if available, otherwise use calculated one
        if metadata.get('checksum_sha256'):
            checksum = metadata['checksum_sha256']
        
        # Split into chunks
        if self.method == "semantic":
            # SemanticChunker works with Document objects
            from langchain_core.documents import Document
            doc = Document(page_content=text, metadata={"source": str(md_path)})
            chunks = self.splitter.split_documents([doc])
            chunk_texts = [chunk.page_content for chunk in chunks]
        else:
            chunk_texts = self.splitter.split_text(text)
        
        # Create chunk records
        chunk_records = []
        for idx, chunk_text in enumerate(chunk_texts, start=1):
            chunk_id = f"{doc_id}_chunk_{idx:04d}"
            
            chunk_record = {
                'doc_id': doc_id,
                'chunk_id': chunk_id,
                'text': chunk_text,
                'site_ids': metadata['site_ids'],
                'concept_ids': metadata['concept_ids'],
                'license': metadata['license'],
                'source_path': str(md_path.resolve()),
                'registry_path': DocumentChunker.REGISTRY_PATH_OUTPUT,
                'checksum_sha256': checksum,
            }
            chunk_records.append(chunk_record)
        
        return chunk_records
    
    def process_all_files(self, output_dir: Optional[Path] = None) -> Dict[str, int]:
        """
        Process all markdown files in ARTICLES_ROOT.
        
        Args:
            output_dir: Optional output directory. If None, saves to same directory as source file.
            
        Returns:
            Dictionary with processing statistics
        """
        stats = {'processed': 0, 'failed': 0, 'total_chunks': 0}
        
        # Find all markdown files matching pattern DOC_paper_*/DOC_paper_*.md
        articles_root = DocumentChunker.ARTICLES_ROOT
        md_files = list(articles_root.glob("DOC_paper_*/DOC_paper_*.md"))
        
        if not md_files:
            print(f"No markdown files found in {articles_root}")
            return stats
        
        print(f"Found {len(md_files)} markdown files to process")
        print(f"Using chunking method: {self.method}")
        
        for md_path in sorted(md_files):
            try:
                print(f"\nProcessing: {md_path.name}")
                
                # Chunk the file
                chunk_records = self.chunk_file(md_path)
                
                # Determine output path
                if output_dir:
                    output_dir.mkdir(parents=True, exist_ok=True)
                    doc_id = self._extract_doc_id_from_path(md_path)
                    output_path = output_dir / f"{doc_id}_chunks.jsonl"
                else:
                    # Save in same directory as source file
                    doc_id = self._extract_doc_id_from_path(md_path)
                    output_path = md_path.parent / f"{doc_id}_chunks.jsonl"
                
                # Write JSONL file
                with open(output_path, 'w', encoding='utf-8') as f:
                    for record in chunk_records:
                        f.write(json.dumps(record, ensure_ascii=False) + '\n')
                
                print(f"  → Created {len(chunk_records)} chunks")
                print(f"  → Saved to: {output_path}")
                
                stats['processed'] += 1
                stats['total_chunks'] += len(chunk_records)
                
            except Exception as e:
                print(f"  ! Failed to process {md_path.name}: {e}")
                stats['failed'] += 1
                import traceback
                traceback.print_exc()
        
        return stats


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Chunk markdown files from Rag_Vault/articles with metadata from registry."
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["recursive", "markdown", "semantic"],
        default="recursive",
        help="Chunking method to use (default: recursive)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Chunk size for recursive and markdown methods (default: 1000)"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Chunk overlap for recursive and markdown methods (default: 200)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for chunks (default: same directory as source file)"
    )
    parser.add_argument(
        "--registry-path",
        type=str,
        default=None,
        help="Path to registry CSV (default: /home/ubuntu/LIA_CARA/src/LIACARA/Rag_Vault/registry/document_master_list.csv)"
    )
    parser.add_argument(
        "--articles-root",
        type=str,
        default=None,
        help="Path to articles root directory (default: /home/ubuntu/LIA_CARA/src/LIACARA/Rag_Vault/articles)"
    )
    
    args = parser.parse_args()
    
    # Override defaults if provided
    if args.registry_path:
        DocumentChunker.REGISTRY_PATH = Path(args.registry_path)
    if args.articles_root:
        DocumentChunker.ARTICLES_ROOT = Path(args.articles_root)
    
    # Create chunker
    chunker = DocumentChunker(
        method=args.method,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    
    # Process files
    output_dir = Path(args.output_dir) if args.output_dir else None
    stats = chunker.process_all_files(output_dir=output_dir)
    
    # Print summary
    print("\n" + "="*60)
    print("Processing Summary")
    print("="*60)
    print(f"Files processed: {stats['processed']}")
    print(f"Files failed: {stats['failed']}")
    print(f"Total chunks created: {stats['total_chunks']}")
    print("="*60)


if __name__ == "__main__":
    main()
