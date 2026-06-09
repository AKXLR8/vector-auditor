"""Download the embedding model + cross-encoder reranker and save as compressed PKLs."""
import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("download_model")

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def download_embedding():
    target = MODELS_DIR / "embedding_model.pkl"
    logger.info("Downloading all-mpnet-base-v2 ...")
    from sentence_transformers import SentenceTransformer
    t0 = time.monotonic()
    model = SentenceTransformer("all-mpnet-base-v2")
    logger.info("Model ready in %.2fs", time.monotonic() - t0)

    import joblib
    t1 = time.monotonic()
    joblib.dump(model, str(target), compress=3)
    raw_mb = target.stat().st_size / (1024 * 1024)
    logger.info("PKL written to %s (%.1f MB) in %.2fs", target, raw_mb, time.monotonic() - t1)

    t2 = time.monotonic()
    loaded = joblib.load(str(target))
    test_vec = loaded.encode(["test sentence"], batch_size=128, show_progress_bar=False)
    logger.info("Verification encode OK (dim=%d) in %.2fs", len(test_vec[0]), time.monotonic() - t2)


def download_reranker():
    target = MODELS_DIR / "reranker.pkl"
    logger.info("Downloading BAAI/bge-reranker-v2-m3 ...")
    from sentence_transformers import CrossEncoder
    t0 = time.monotonic()
    model = CrossEncoder("BAAI/bge-reranker-v2-m3")
    logger.info("Reranker ready in %.2fs", time.monotonic() - t0)

    import joblib
    t1 = time.monotonic()
    joblib.dump(model, str(target), compress=3)
    raw_mb = target.stat().st_size / (1024 * 1024)
    logger.info("PKL written to %s (%.1f MB) in %.2fs", target, raw_mb, time.monotonic() - t1)


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    download_embedding()
    download_reranker()
    logger.info("All models downloaded and cached.")


if __name__ == "__main__":
    main()
