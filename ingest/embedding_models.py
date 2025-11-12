# embedding_models.py
from typing import List
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from sentence_transformers import SentenceTransformer
import open_clip

# ------------------ CONFIG ------------------
TEXT_DIM = 384
IMG_DIM = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEXT_BATCH = 256
IMG_BATCH = 64

# ------------------ MODELS ------------------
print(f"Loading models to {DEVICE}...")
text_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=DEVICE).eval()
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="openai", device=DEVICE
)
clip_model.eval()
print("Models loaded.")

# ------------------ EMBEDDING FUNCTIONS ------------------
def embed_texts(texts: List[str]) -> np.ndarray:
    """Embeds a list of texts, normalizing them for cosine similarity."""
    with torch.no_grad():
        embs = text_model.encode(
            texts, 
            batch_size=TEXT_BATCH, 
            convert_to_numpy=True, 
            normalize_embeddings=False, # We normalize manually
            show_progress_bar=False
        )
    # Normalize for cosine-friendly storage
    n = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
    return (embs / n).astype(np.float32)

def embed_images(paths: List[Path]) -> np.ndarray:
    """Embeds a list of image paths, handling errors and returning normalized vectors."""
    tensors = []
    kept_idx = []
    for i, p in enumerate(paths):
        try:
            img = Image.open(p).convert("RGB")
            tensors.append(clip_preprocess(img))
            kept_idx.append(i)
        except Exception:
            tensors.append(None) # Add a placeholder for failed images

    if not kept_idx:
        return np.zeros((len(paths), IMG_DIM), dtype=np.float32)

    batch = torch.stack([t for t in tensors if t is not None]).to(DEVICE)
    with torch.no_grad():
        feats = clip_model.encode_image(batch)
        feats = feats / feats.norm(dim=-1, keepdim=True) # Normalize
    feats = feats.detach().cpu().numpy().astype(np.float32)

    # Restore original order, filling in zeros for failed loads
    out = np.zeros((len(paths), IMG_DIM), dtype=np.float32)
    j = 0
    for i in range(len(paths)):
        if tensors[i] is not None:
            out[i] = feats[j]
            j += 1
    return out
