"""Pre-download & pickle models during build so startup is instant."""
import logging
import shutil
from pathlib import Path

import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("download_models")

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def download_embedding_model():
    from sentence_transformers import SentenceTransformer

    logger.info("Downloading all-mpnet-base-v2...")
    model = SentenceTransformer("all-mpnet-base-v2")
    path = MODELS_DIR / "embedding_model.pkl"
    joblib.dump(model, str(path), compress=True)
    logger.info("Saved -> %s (%.1f MB)", path, path.stat().st_size / 1e6)


def download_reranker():
    from sentence_transformers import CrossEncoder

    logger.info("Downloading BAAI/bge-reranker-v2-m3...")
    model = CrossEncoder("BAAI/bge-reranker-v2-m3")
    path = MODELS_DIR / "reranker.pkl"
    joblib.dump(model, str(path), compress=True)
    logger.info("Saved -> %s (%.1f MB)", path, path.stat().st_size / 1e6)


def main():
    if MODELS_DIR.exists():
        shutil.rmtree(MODELS_DIR)
    MODELS_DIR.mkdir(parents=True)
    download_embedding_model()
    download_reranker()
    logger.info("Done.")


if __name__ == "__main__":
    main()
