#!/bin/sh
# Index the catalog on first start (or when the embedding model changed),
# then serve the web UI.
set -e

python -m src.indexer --if-needed

exec gunicorn --bind "0.0.0.0:${FLASK_PORT:-8000}" --workers 1 --threads 4 \
    --timeout 120 --access-logfile - "src.web_app:app"
