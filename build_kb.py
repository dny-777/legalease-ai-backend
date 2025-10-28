import json
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

KB_FILE = "kb/kb.json"
INDEX_FILE = "kb/index.faiss"
MAPPING_FILE = "kb/index_mapping.json"

def build_kb():
    if not os.path.exists(KB_FILE):
        logger.error(f"{KB_FILE} not found. Please add it first.")
        return

    logger.info("Starting Knowledge Base build...")
    
    with open(KB_FILE, "r", encoding="utf-8") as f:
        kb_data = json.load(f)

    if not kb_data:
        logger.warning("KB data is empty. Skipping index build.")
        return

    logger.info("Loading SentenceTransformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    texts = [item['text'] for item in kb_data]
    ids = [item['id'] for item in kb_data]

    logger.info(f"Encoding {len(texts)} documents...")
    embeddings = model.encode(texts, show_progress_bar=True)
    
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    id_map = faiss.IndexIDMap(index)
    
    faiss_ids = np.array(range(len(ids)))
    
    id_map.add_with_ids(embeddings, faiss_ids)
    
    logger.info(f"Saving FAISS index to {INDEX_FILE}")
    faiss.write_index(id_map, INDEX_FILE)
    
    mapping = {i: ids[i] for i in range(len(ids))}
    with open(MAPPING_FILE, "w") as f:
        json.dump(mapping, f)

    logger.info("Knowledge Base build complete.")

if __name__ == "__main__":
    build_kb()