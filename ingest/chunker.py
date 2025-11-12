# chunker.py
#!/usr/bin/env python3
"""
Chunk markdown files from Rag_Vault/articles/DOC_paper_* directories.
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownTextSplitter,
)
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

# Import from our new utility module
import utils

class DocumentChunker:
    """Chunk markdown files with metadata from registry CSV."""
    
    _LIACARA_ROOT: Optional[Path] = None
    REGISTRY_PATH_OUTPUT = "/LIACARA/Rag_Vault/registry/document_master_list.csv"
    
    @classmethod
    def get_liacara_root(cls) -> Path:
        """Get or find the LIACARA root directory."""
        if cls._LIACARA_ROOT is None:
            cls._LIACARA_ROOT = utils.find_liacara_root()
        return cls._LIACARA_ROOT
    
    @classmethod
    def get_registry_path(cls) -> Path:
        """Get the registry CSV path relative to LIACARA root."""
        return cls.get_liacara_root() / "Rag_Vault" / "registry" / "document_master_list.csv"
    
    @classmethod
    def get_articles_root(cls) -> Path:
        """Get the articles root path relative to LIACARA root."""
        return cls.get_liacara_root() / "Rag_Vault" / "articles"
    
    def __init__(self, method: str = "recursive", chunk_size: int = 1000, chunk_overlap: int = 200, embeddings=None):
        self.method = method.lower()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embeddings = embeddings
        self.registry_data = utils.load_document_registry(self.get_registry_path())
        self.splitter = self._create_splitter()
        
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
            if self.embeddings is None:
                try:
                    self.embeddings = OpenAIEmbeddings()
                except Exception as e:
                    raise RuntimeError(
                        f"SemanticChunker requires embeddings. "
                        f"Either provide embeddings parameter or set OPENAI_API_KEY. Error: {e}"
                    )
            return SemanticChunker(
                embeddings=self.embeddings,
                buffer_size=1,
            )
        else:
            raise ValueError(f"Unknown method: {self.method}. Choose from: recursive, markdown, semantic")
    
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
        """Chunks a single markdown file and returns list of chunk dictionaries."""
        text = md_path.read_text(encoding='utf-8', errors='ignore')
        checksum = utils.calculate_sha256(md_path)
        doc_id = utils.extract_doc_id_from_path(md_path)
        short_id = str(doc_id).replace("DOC_paper_", "")
        metadata = self._get_doc_metadata(doc_id)
        
        # Use registry checksum if available, otherwise use calculated one
        file_checksum = metadata.get('checksum_sha256') or checksum
        
        # Split into chunks
        if self.method == "semantic":
            from langchain_core.documents import Document
            doc = Document(page_content=text, metadata={"source": str(md_path)})
            chunks = self.splitter.split_documents([doc])
            chunk_texts = [chunk.page_content for chunk in chunks]
        else:
            chunk_texts = self.splitter.split_text(text)
        
        # Create chunk records
        chunk_records = []
        for idx, chunk_text in enumerate(chunk_texts, start=1):
            chunk_id = f"CHUNK_{short_id}_{idx:04d}"
            
            chunk_record = {
                'doc_id': doc_id,
                'chunk_id': chunk_id,
                'text': chunk_text,
                'site_ids': metadata['site_ids'],
                'concept_ids': metadata['concept_ids'],
                'license': metadata['license'],
                'source_path': str(md_path.resolve()),
                'registry_path': DocumentChunker.REGISTRY_PATH_OUTPUT,
                'checksum_sha256': file_checksum,
            }
            chunk_records.append(chunk_record)
        
        return chunk_records
    
    def process_all_files(self, output_dir: Optional[Path] = None) -> Dict[str, int]:
        """Process all markdown files in ARTICLES_ROOT."""
        stats = {'processed': 0, 'failed': 0, 'total_chunks': 0}
        
        articles_root = DocumentChunker.get_articles_root()
        md_files = list(articles_root.glob("DOC_paper_*/DOC_paper_*.md"))
        
        if not md_files:
            print(f"No markdown files found in {articles_root}")
            return stats
        
        print(f"Found {len(md_files)} markdown files to process")
        print(f"Using chunking method: {self.method}")
        
        for md_path in sorted(md_files):
            try:
                print(f"\nProcessing: {md_path.name}")
                chunk_records = self.chunk_file(md_path)
                
                if output_dir:
                    output_dir.mkdir(parents=True, exist_ok=True)
                    doc_id = utils.extract_doc_id_from_path(md_path)
                    output_path = output_dir / f"{doc_id}_chunks.jsonl"
                else:
                    doc_id = utils.extract_doc_id_from_path(md_path)
                    output_path = md_path.parent / f"{doc_id}_chunks.jsonl"
                
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
        "--liacara-root",
        type=str,
        default=None,
        help="Path to LIACARA root directory (default: auto-detected)"
    )
    
    args = parser.parse_args()
    
    if args.liacara_root:
        DocumentChunker._LIACARA_ROOT = Path(args.liacara_root).resolve()
    
    print(f"LIACARA root: {DocumentChunker.get_liacara_root()}")
    print(f"Registry path: {DocumentChunker.get_registry_path()}")
    print(f"Articles root: {DocumentChunker.get_articles_root()}")
    
    chunker = DocumentChunker(
        method=args.method,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    
    output_dir = Path(args.output_dir) if args.output_dir else None
    stats = chunker.process_all_files(output_dir=output_dir)
    
    print("\n" + "="*60)
    print("Processing Summary")
    print("="*60)
    print(f"Files processed: {stats['processed']}")
    print(f"Files failed: {stats['failed']}")
    print(f"Total chunks created: {stats['total_chunks']}")
    print("="*60)

if __name__ == "__main__":
    main()
