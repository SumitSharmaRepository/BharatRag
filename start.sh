#!/bin/bash
echo "Starting BharatRAG on port: $PORT"
exec uvicorn api:app --host 0.0.0.0 --port "${PORT:-8000}"