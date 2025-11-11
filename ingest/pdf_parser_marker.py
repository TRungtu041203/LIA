import argparse
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import List, Tuple, Dict


IMAGE_MD_PATTERN = re.compile(r'!\[(?P<alt>[^\]]*)\]\((?P<src>[^)]+)\)')
TABLE_SEPARATOR_PATTERN = re.compile(
    r'^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$'
)

def run_cmd(cmd: List[str]) -> Tuple[int, str, str]:
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = proc.communicate()
    return proc.returncode, out, err

def run_marker_convert(pdf_path: Path, out_dir: Path) -> Path:
    """
    Run Marker on a single PDF into out_dir.
    Returns the path to the created Markdown file.
    Tries several CLI variants for portability.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    candidate_cmds = [
        ["marker", "convert", str(pdf_path), "--output-dir", str(out_dir)],
        ["marker", str(pdf_path), "--output-dir", str(out_dir)],
        ["python", "-m", "marker", "convert", str(pdf_path), "--output-dir", str(out_dir)],
        ["python3", "-m", "marker", "convert", str(pdf_path), "--output-dir", str(out_dir)],
    ]

    last_err = ""
    for cmd in candidate_cmds:
        code, out, err = run_cmd(cmd)
        if code == 0:
            # Find the newest .md in out_dir (Marker typically writes exactly one)
            md_files = sorted(out_dir.rglob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
            if md_files:
                return md_files[0]
            else:
                last_err = f"Marker ran but no .md found in {out_dir}"
        else:
            last_err = err

    raise RuntimeError(
        f"Failed to run Marker. Last error:\n{last_err}\n"
        "Check that 'marker' is installed (pip install marker-pdf) and on PATH."
    )

def is_table_block(lines: List[str], start_idx: int) -> Tuple[bool, int]:
    """
    Heuristically detect a GitHub-style Markdown table starting at start_idx.
    Returns (is_table, end_idx_exclusive).
    A table block is contiguous lines that start with '|' (or have multiple pipes)
    and contain a separator line like: | --- | :---: | ---: |
    """
    if start_idx >= len(lines):
        return False, start_idx

    # Quick pre-check: must look like a table row (pipes)
    first = lines[start_idx].rstrip("\n")
    if "|" not in first:
        return False, start_idx

    # Collect contiguous pipe-y lines
    i = start_idx
    block = []
    while i < len(lines) and ("|" in lines[i]):
        block.append(lines[i].rstrip("\n"))
        i += 1

    # Must contain a separator line
    if any(bool(TABLE_SEPARATOR_PATTERN.match(row)) for row in block):
        return True, i
    return False, start_idx

def extract_images_from_md(md_text: str, md_dir: Path) -> List[Dict]:
    images = []
    for m in IMAGE_MD_PATTERN.finditer(md_text):
        alt = m.group("alt")
        src = m.group("src")
        # Normalize path; Marker typically emits relative paths
        img_path = (md_dir / src).resolve()
        images.append({"alt": alt, "src": str(img_path)})
    return images

def markdown_table_to_csv_rows(table_lines: List[str]) -> List[List[str]]:
    """
    Very simple Markdown table parser:
    - strips leading/trailing '|' and splits by '|'
    - skips the second line if it's a separator row
    - trims cell whitespace
    Note: does not handle escaped pipes inside cells.
    """
    rows = []
    for idx, raw in enumerate(table_lines):
        line = raw.strip()
        if not line:
            continue
        # skip separator line
        if TABLE_SEPARATOR_PATTERN.match(line):
            continue
        # remove leading/trailing pipes then split
        if line.startswith("|"):
            line = line[1:]
        if line.endswith("|"):
            line = line[:-1]
        cells = [c.strip() for c in line.split("|")]
        rows.append(cells)
    return rows

def write_csv(path: Path, rows: List[List[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            # naive CSV—quote if needed
            escaped = []
            for cell in r:
                cell2 = cell.replace('"', '""')
                if any(ch in cell2 for ch in [",", '"', "\n"]):
                    escaped.append(f'"{cell2}"')
                else:
                    escaped.append(cell2)
            f.write(",".join(escaped) + "\n")

def split_md_text_tables_images(md_path: Path) -> Tuple[str, List[List[str]], List[Dict]]:
    """
    Returns:
      text_only (images & tables stripped),
      tables_raw (list of list-of-lines),
      images_meta (list of dicts with alt/src)
    """
    text = md_path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()

    # Collect table blocks
    i = 0
    table_blocks = []
    table_spans = []
    while i < len(lines):
        is_tbl, end_i = is_table_block(lines, i)
        if is_tbl:
            table_blocks.append(lines[i:end_i])
            table_spans.append((i, end_i))
            i = end_i
        else:
            i += 1

    # Strip out table lines
    table_line_idxs = set()
    for s, e in table_spans:
        for k in range(s, e):
            table_line_idxs.add(k)

    # Strip out image lines
    image_line_idxs = set()
    for idx, line in enumerate(lines):
        if IMAGE_MD_PATTERN.search(line):
            image_line_idxs.add(idx)

    # Produce text-only (keep other markdown)
    text_only_lines = [
        ln for idx, ln in enumerate(lines)
        if idx not in table_line_idxs and idx not in image_line_idxs
    ]
    text_only = "\n".join(text_only_lines)

    # Images metadata
    images_meta = extract_images_from_md(text, md_path.parent)

    return text_only, table_blocks, images_meta

def copy_images(images_meta: List[Dict], dest_dir: Path) -> List[str]:
    copied = []
    dest_dir.mkdir(parents=True, exist_ok=True)
    for im in images_meta:
        src = Path(im["src"])
        if src.exists():
            target = dest_dir / src.name
            # Avoid overwriting with identical names; add counter if needed
            base = target.stem
            ext = target.suffix
            c = 1
            while target.exists():
                target = dest_dir / f"{base}_{c}{ext}"
                c += 1
            shutil.copy2(src, target)
            copied.append(str(target.resolve()))
    return copied

def process_single_pdf(pdf_path: Path, out_root: Path) -> Dict:
    pdf_stem = pdf_path.stem
    work_dir = out_root / pdf_stem
    work_dir.mkdir(parents=True, exist_ok=True)

    # 1) Run Marker
    md_path = run_marker_convert(pdf_path, work_dir)

    # 2) Split MD into text/tables/images
    text_only, table_blocks, images_meta = split_md_text_tables_images(md_path)

    # 3) Save text-only
    text_path = work_dir / "text_only.txt"
    text_path.write_text(text_only, encoding="utf-8")

    # 4) Save tables (.md and .csv)
    tables_dir = work_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    table_md_paths = []
    table_csv_paths = []
    for idx, tbl in enumerate(table_blocks, start=1):
        md_out = tables_dir / f"table_{idx}.md"
        md_out.write_text("\n".join(tbl) + "\n", encoding="utf-8")
        table_md_paths.append(str(md_out.resolve()))
        # Make a simple CSV
        rows = markdown_table_to_csv_rows(tbl)
        if rows:
            csv_out = tables_dir / f"table_{idx}.csv"
            write_csv(csv_out, rows)
            table_csv_paths.append(str(csv_out.resolve()))

    # 5) Copy images to local subdir
    images_dir = work_dir / "images"
    image_paths = copy_images(images_meta, images_dir)

    # 6) Build record
    record = {
        "pdf": str(pdf_path.resolve()),
        "bundle_dir": str(work_dir.resolve()),
        "markdown": str(md_path.resolve()),
        "text_only": str(text_path.resolve()),
        "tables_markdown": table_md_paths,
        "tables_csv": table_csv_paths,
        "images": image_paths,
        "num_tables": len(table_md_paths),
        "num_images": len(image_paths),
    }
    return record

def main():
    parser = argparse.ArgumentParser(description="Batch parse PDFs with Marker into text/tables/images.")
    parser.add_argument("--pdf_dir", type=str, default="pdf",
                        help="Folder containing PDF files.")
    parser.add_argument("--out_dir", type=str, default="parsed",
                        help="Output root folder.")
    args = parser.parse_args()

    pdf_dir = Path(args.pdf_dir)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    pdfs = sorted([p for p in pdf_dir.glob("**/*.pdf") if p.is_file()])
    if not pdfs:
        print(f"No PDFs found in {pdf_dir.resolve()}")
        return

    index = []
    for pdf in pdfs:
        print(f"[+] Processing: {pdf}")
        try:
            rec = process_single_pdf(pdf, out_root)
            index.append(rec)
            print(f"    → OK: {rec['bundle_dir']}")
        except Exception as e:
            print(f"    ! Failed on {pdf.name}: {e}")

    # Write dataset index
    idx_path = out_root / "dataset_index.json"
    idx_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote index: {idx_path.resolve()}")
    print(f"Done. {len(index)} PDFs processed successfully.")

if __name__ == "__main__":
    main()
