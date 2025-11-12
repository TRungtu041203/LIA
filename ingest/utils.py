# utils.py
import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Any

def find_liacara_root(start_path: Optional[Path] = None) -> Path:
    """
    Find the LIACARA root directory by searching upward from the script location.
    """
    if start_path is None:
        # Start from the directory containing this script
        start_path = Path.cwd() # Use current working dir as a robust default
    
    current = Path(start_path).resolve()
    
    # Search upward for LIACARA folder
    for parent in [current] + list(current.parents):
        liacara_path = parent / "LIACARA"
        if liacara_path.exists() and liacara_path.is_dir():
            return liacara_path.resolve()
    
    # Alternative: search for LIACARA in current directory or parents
    for parent in [current] + list(current.parents):
        if parent.name == "LIACARA" and parent.is_dir():
            return parent.resolve()
    
    raise FileNotFoundError(
        f"Could not find LIACARA directory. Searched from {start_path} upward. "
        f"Please ensure the LIACARA folder exists in the project structure."
    )

def calculate_sha256(file_path: Path) -> str:
    """Calculate SHA256 checksum of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def extract_doc_id_from_path(md_path: Path) -> str:
    """Extract doc_id from markdown file path (e.g., DOC_paper_01.md -> DOC_paper_01)."""
    return md_path.stem

def read_jsonl_from_articles_root(articles_root: Path):
    """Read all DOC_paper_*_chunks.jsonl files from DOC_paper_* directories."""
    for doc_dir in sorted(articles_root.glob("DOC_paper_*")):
        if not doc_dir.is_dir():
            continue
        chunks_file = doc_dir / f"{doc_dir.name}_chunks.jsonl"
        if chunks_file.exists():
            with open(chunks_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        yield json.loads(line)

def list_images(root: str) -> List[Path]:
    """Finds all common image files recursively."""
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
    rootp = Path(root)
    return [q for q in sorted(rootp.rglob("*")) if q.suffix.lower() in exts] if rootp.exists() else []

def load_media_registry(csv_path: Optional[str]) -> Dict[str, Dict]:
    """Load media registry CSV and index by media_id, filename, and stem."""
    if not csv_path or not Path(csv_path).exists():
        return {}
    reg: Dict[str, Dict] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            media_id = (row.get("media_id") or "").strip()
            if media_id:
                reg[media_id] = row
            path = (row.get("path") or "").strip()
            if path:
                filename = Path(path).name
                if filename and filename not in reg:
                    reg[filename] = row
                stem = Path(path).stem
                if stem and stem not in reg:
                    reg[stem] = row
    return reg

def load_document_registry(registry_path: Path) -> Dict[str, Dict[str, Any]]:
    """Load document metadata from registry CSV."""
    registry = {}
    if not registry_path.exists():
        raise FileNotFoundError(f"Registry not found: {registry_path}")
    
    with open(registry_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            doc_id = row['doc_id']
            site_ids = to_tags(row.get('site_ids'))
            concept_ids = to_tags(row.get('concept_ids'))
            
            registry[doc_id] = {
                'site_ids': site_ids,
                'concept_ids': concept_ids,
                'license': row.get('license', ''),
                'checksum_sha256': row.get('checksum_sha256', ''),
            }
    return registry

def to_tags(s: Optional[str]):
    """Convert pipe, semicolon, or comma-separated string to list."""
    if not s:
        return []
    for sep in ["|", ";", ","]:
        if sep in s:
            return [x.strip() for x in s.split(sep) if x.strip()]
    return [s.strip()] if s.strip() else []

def to_int(x):
    """Safely convert a value to an integer or None."""
    try:
        return int(x) if x not in (None, "") else None
    except (ValueError, TypeError):
        return None
