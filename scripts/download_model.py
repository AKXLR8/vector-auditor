"""Download the embedding model and save as compressed PKL for fast startup."""
import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("download_model")

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def main() -> None:
    t0 = time.monotonic()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    target = MODELS_DIR / "embedding_model.pkl"

    logger.info("Downloading all-MiniLM-L6-v2 ...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    logger.info("Model ready in %.2fs", time.monotonic() - t0)

    import joblib
    t1 = time.monotonic()
    joblib.dump(model, str(target), compress=3)
    raw_mb = target.stat().st_size / (1024 * 1024)
    logger.info("PKL written to %s (%.1f MB) in %.2fs", target, raw_mb, time.monotonic() - t1)

    # Verify it loads back
    t2 = time.monotonic()
    loaded = joblib.load(str(target))
    test_vec = loaded.encode(["test sentence"], batch_size=128, show_progress_bar=False)
    logger.info("Verification encode OK (dim=%d) in %.2fs", len(test_vec[0]), time.monotonic() - t2)

    logger.info("Total: %.2fs — PKL ready", time.monotonic() - t0)


if __name__ == "__main__":
    main()
