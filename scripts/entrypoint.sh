#!/bin/sh
set -e

echo "[entrypoint] Running database migrations..."
alembic upgrade head

echo "[entrypoint] Migrations complete, starting gunicorn..."
exec gunicorn src.api.main:app --config gunicorn_conf.py
