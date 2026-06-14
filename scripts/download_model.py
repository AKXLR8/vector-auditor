"""Download the embedding model + cross-encoder reranker and save as compressed PKLs.
Memory-safe: frees each model before loading the next."""
import gc
import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("download_model")

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def download_embedding():
    target = MODELS_DIR / "embedding_model.pkl"
    if target.exists():
        logger.info("Embedding PKL already exists, skipping")
        return
    logger.info("Downloading all-MiniLM-L6-v2 ...")
    from sentence_transformers import SentenceTransformer
    t0 = time.monotonic()
    model = SentenceTransformer("all-MiniLM-L6-v2")
    logger.info("Model ready in %.2fs", time.monotonic() - t0)

    import joblib
    t1 = time.monotonic()
    joblib.dump(model, str(target), compress=3)
    raw_mb = target.stat().st_size / (1024 * 1024)
    logger.info("PKL written to %s (%.1f MB) in %.2fs", target, raw_mb, time.monotonic() - t1)

    # Verify loads back
    t2 = time.monotonic()
    loaded = joblib.load(str(target))
    test_vec = loaded.encode(["test sentence"], batch_size=128, show_progress_bar=False)
    logger.info("Verification encode OK (dim=%d) in %.2fs", len(test_vec[0]), time.monotonic() - t2)

    # Free memory before next download
    del model, loaded, test_vec
    gc.collect()


def download_reranker():
    target = MODELS_DIR / "reranker.pkl"
    if target.exists():
        logger.info("Reranker PKL already exists, skipping")
        return
    # Download raw HF snapshot (weight files only, no model load into RAM)
    # so CrossEncoder finds it in cache at runtime instead of downloading.
    from huggingface_hub import constants, snapshot_download
    cache_dir = str(MODELS_DIR / "hf_cache")
    logger.info("Downloading BAAI/bge-reranker-base snapshot to %s ...", cache_dir)
    t0 = time.monotonic()
    snapshot_download("BAAI/bge-reranker-base", cache_dir=cache_dir)
    total_mb = sum(f.stat().st_size for f in Path(cache_dir).rglob("*") if f.is_file()) / (1024 * 1024)
    logger.info("Reranker snapshot downloaded (%.1f MB) in %.2fs", total_mb, time.monotonic() - t0)


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    download_embedding()
    download_reranker()
    logger.info("All models downloaded and cached.")


if __name__ == "__main__":
    main()
