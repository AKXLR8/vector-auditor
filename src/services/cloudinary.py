"""Cloudinary storage backend. Optional — only used when CLOUDINARY_* env vars are set."""
import io
import logging
import os
from typing import Optional

from .secrets import get_secret

logger = logging.getLogger("rga_auditor.cloudinary")


class CloudinaryStorage:
    def __init__(self) -> None:
        self._enabled = False
        cloud_name = get_secret("CLOUDINARY_CLOUD_NAME") or os.getenv("CLOUDINARY_CLOUD_NAME")
        api_key = get_secret("CLOUDINARY_API_KEY") or os.getenv("CLOUDINARY_API_KEY")
        api_secret = get_secret("CLOUDINARY_API_SECRET") or os.getenv("CLOUDINARY_API_SECRET")
        if not (cloud_name and api_key and api_secret):
            logger.info("Cloudinary not configured — using local disk storage")
            return
        try:
            import cloudinary
            import cloudinary.uploader
            cloudinary.config(cloud_name=cloud_name, api_key=api_key, api_secret=api_secret, secure=True)
            self._uploader = cloudinary.uploader
            self._enabled = True
            logger.info("Cloudinary storage enabled (cloud=%s)", cloud_name)
        except Exception as e:
            logger.warning("Cloudinary init failed: %s", e)

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def upload(self, content: bytes, public_id: str, resource_type: str = "raw") -> dict:
        if not self._enabled:
            raise RuntimeError("Cloudinary not configured")
        return self._uploader.upload(
            io.BytesIO(content),
            public_id=public_id,
            resource_type=resource_type,
            overwrite=True,
        )

    def get_url(self, public_id: str, resource_type: str = "raw") -> str:
        if not self._enabled:
            return ""
        import cloudinary
        return cloudinary.CloudinaryResource(public_id=public_id, resource_type=resource_type).build_url()

    def get_signed_url(self, public_id: str, resource_type: str = "image", format: str = "pdf", version: str = None) -> str:
        if not self._enabled:
            return ""
        from cloudinary.utils import cloudinary_url
        opts = dict(
            resource_type=resource_type,
            type="upload",
            format=format,
            sign_url=True,
        )
        if version:
            opts["version"] = version
        url, _ = cloudinary_url(public_id, **opts)
        return url


_storage: Optional[CloudinaryStorage] = None


def get_cloudinary() -> CloudinaryStorage:
    global _storage
    if _storage is None:
        _storage = CloudinaryStorage()
    return _storage
