web: gunicorn src.api.main:app --config gunicorn_conf.py --bind 0.0.0.0:$PORT
release: alembic upgrade head
