"""Secrets resolution order:
1. AWS Secrets Manager (if AWS_REGION set and secret exists)
2. Environment variable
3. None
"""
import logging
import os
from typing import Optional

logger = logging.getLogger("rga_auditor.secrets")

_aws_client = None


def _get_aws_client():
    global _aws_client
    if _aws_client is not None:
        return _aws_client
    try:
        import boto3
        region = os.getenv("AWS_REGION", "us-east-1")
        _aws_client = boto3.client("secretsmanager", region_name=region)
    except Exception as e:
        logger.debug("AWS Secrets Manager unavailable: %s", e)
        _aws_client = False  # sentinel: don't try again
    return _aws_client or None


def get_secret(key: str) -> Optional[str]:
    client = _get_aws_client()
    if client:
        try:
            r = client.get_secret_value(SecretId=key)
            return r.get("SecretString")
        except Exception as e:
            logger.debug("AWS Secrets Manager lookup failed for %s: %s", key, e)
    return os.getenv(key)
