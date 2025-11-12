# ingest.py
import numpy as np
from pathlib import Path
from tqdm import tqdm
from qdrant_client import models
from typing import List, Dict, Any

# Import from our new modules
from vector_store import VectorStore
import utils
import embedding_models as em

# ------------------ CONFIG ------------------
_LIACARA_ROOT = utils.find_liacara_root()
QDRANT_URL = "http://localhost:6333"

# Paths relative to LIACARA root
ARTICLES_ROOT = _LIACARA_ROOT / "Rag_Vault" / "articles"
IMAGES_DIR = str(_LIACARA_ROOT / "Media_Vault" / "images")
MEDIA_REGISTRY_CSV = str(_LIACARA_ROOT / "Media_Vault" / "registry" / "media_registry.csv")
REGISTRY_PATH_OUTPUT = "/LIACARA/Media_Vault/registry/media_registry.csv"

# ------------------ MAIN PIPELINE ------------------
def main():
    vs = VectorStore(url=QDRANT_URL)

    # 1) CREATE collections
    print("Creating 'rag_text_chunks' collection...")
    vs.create_or_recreate_collection(
        "rag_text_chunks",
        vectors=(em.TEXT_DIM, models.Distance.COSINE),
        on_disk_payload=True,
        force=True,
    )
    print("Creating 'media_assets' collection...")
    vs.create_or_recreate_collection(
        "media_assets",
        vectors={
            "image":  (em.IMG_DIM,   models.Distance.COSINE),
            "caption":(em.TEXT_DIM,  models.Distance.COSINE),
        },
        on_disk_payload=True,
        force=True,
    )

    # 2) Add payload indexes
    print("Creating payload indexes...")
    for fld in ["doc_id", "page", "is_caption", "figure_id", "site_ids", "concept_ids", "license"]:
        try: vs.create_payload_index("rag_text_chunks", fld)
        except Exception: pass
    #for fld in ["media_id", "site_ids", "concept_ids", "license", "sensitivity", "asset_type", "parent_doc_id", "page", "figure_id"]:
    for fld in ["media_id", "site_ids", "concept_ids", "license", "sensitivity"]:
        try: vs.create_payload_index("media_assets", fld)
        except Exception: pass

    # 3) INGEST TEXT
    text_ids, text_vecs, text_payloads = [], [], []
    buffer_texts, buffer_meta = [], []

    print(f"Reading JSONL text chunks from {ARTICLES_ROOT}…")
    for row in utils.read_jsonl_from_articles_root(ARTICLES_ROOT):
        chunk_id = row.get("chunk_id") or row.get("id")
        text = (row.get("text") or "").strip()
        if not chunk_id or not text:
            continue
        
        payload = {k: v for k, v in row.items() if v is not None}
        buffer_texts.append(text)
        buffer_meta.append(payload) #List dict of metadata 

        if len(buffer_texts) >= em.TEXT_BATCH:
            emb = em.embed_texts(buffer_texts)
            for i in range(len(buffer_texts)):
                text_ids.append(i)
                text_vecs.append(emb[i].tolist())
                text_payloads.append(buffer_meta[i])
            buffer_texts.clear(); buffer_meta.clear()

    if buffer_texts: # Process remainder
        emb = em.embed_texts(buffer_texts)
        for i in range(len(buffer_texts)):
            #text_ids.append(buffer_meta[i]["chunk_id"])
            text_ids.append(i+ len(text_ids))  # Use sequential IDs if chunk_id missing
            text_vecs.append(emb[i].tolist())
            text_payloads.append(buffer_meta[i]) #Get payloads from buffer_meta cause already processed during chunking

    print(f"Upserting {len(text_ids)} text points…")
    vs.upsert_points("rag_text_chunks", text_ids, text_vecs, text_payloads, batch_size=256)

    # 4) INGEST IMAGES
    registry = utils.load_media_registry(MEDIA_REGISTRY_CSV)
    img_paths = utils.list_images(IMAGES_DIR)
    print(f"Found {len(img_paths)} images in {IMAGES_DIR}")

    if not img_paths:
        print("No images found to ingest.")
    else:
        # Embed images in batches
        image_vectors_list: List[np.ndarray] = []
        for start in tqdm(range(0, len(img_paths), em.IMG_BATCH), desc="Embedding images"):
            batch_paths = img_paths[start:start + em.IMG_BATCH]
            vec = em.embed_images(batch_paths)
            image_vectors_list.append(vec)
        
        image_vectors = np.vstack(image_vectors_list).astype(np.float32)

        # Embed captions
        caption_texts = [
            (registry.get(p.name) or registry.get(p.stem) or {}).get("source_ref", "").strip()
            for p in img_paths
        ]
        caption_vectors = em.embed_texts([t or "" for t in caption_texts])

        # Build and upsert points
        media_ids, media_vecs, media_payloads = [], [], []
        for i, p in enumerate(img_paths):
            row = registry.get(p.name) or registry.get(p.stem) or {}
            # media_id = row.get("media_id") or p.stem
            media_id = i 
            vec_dict = {"image": image_vectors[i].tolist()}
            if caption_texts[i]: # Only add caption vector if text exists
                vec_dict["caption"] = caption_vectors[i].tolist()

            source_path = row.get("path") or str(p.resolve())
            
            payload = {
                "media_id": media_id,
                "site_ids": utils.to_tags(row.get("site_ids")),
                "concept_ids": utils.to_tags(row.get("concept_ids")),
                "source_path": source_path,
                "registry_path": REGISTRY_PATH_OUTPUT,
                "license": row.get("license") or "",
                "checksum_sha256": row.get("checksum_sha256") or "",
                "sensitivity": row.get("sensitivity") or "",
                "asset_type": row.get("media_type") or "image",
                "parent_doc_id": row.get("parent_ids") or None,
                "image_uri": p.resolve().as_uri(),
                "caption_text": caption_texts[i] or "",
                "width": utils.to_int(row.get("width")),
                "height": utils.to_int(row.get("height")),
            }
            media_ids.append(media_id)
            media_vecs.append(vec_dict)
            media_payloads.append(payload)

        print(f"Upserting {len(media_ids)} media points…")
        vs.upsert_points("media_assets", media_ids, media_vecs, media_payloads, batch_size=256)

    # 5) Quick sanity
    print("--- Ingestion Complete ---")
    print(f"rag_text_chunks: {vs.count('rag_text_chunks')} points")
    print(f"media_assets:    {vs.count('media_assets')} points")

    # 6) Optional guards
    try:
        if len(text_vecs) > 0:
            vs.assert_vector_dim("rag_text_chunks", None, text_vecs[0])
        if len(media_vecs) > 0:
            vs.assert_vector_dim("media_assets", "image", media_vecs[0]["image"])
            if "caption" in media_vecs[0]:
                vs.assert_vector_dim("media_assets", "caption", media_vecs[0]["caption"])
        print("Vector dimensions verified.")
    except Exception as e:
        print(f"Warning: Could not verify vector dimensions: {e}")


if __name__ == "__main__":
    main()
