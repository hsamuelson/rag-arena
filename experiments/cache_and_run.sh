#!/bin/bash
# Cache snowflake embeddings, then run remaining composability strategies.
set -e

export SSL_CERT_FILE=$(.venv/bin/python -c "import certifi; print(certifi.where())")
export REQUESTS_CA_BUNDLE=$SSL_CERT_FILE

echo "=== Phase 1: Caching embeddings ==="
.venv/bin/python experiments/cache_embeddings.py "Snowflake/snowflake-arctic-embed-l"

echo ""
echo "=== Phase 2: Running remaining composability strategies ==="
.venv/bin/python experiments/run_composability_remaining.py
