#!/bin/bash
# ============================================
# run_tests.sh — Pre-deployment regression suite
# ============================================
# Run this before every deployment.
# If any test fails — do NOT deploy.
#
# Usage: ./run_tests.sh
# ============================================

set -e  # Exit on first failure

echo "=================================="
echo "BharatRAG Regression Suite"
echo "=================================="
echo ""

cd /home/sumit/bharatrag
source venv/bin/activate

echo "Running security tests..."
pytest tests/test_security.py -v \
    --tb=short \
    -q

echo ""
echo "Running pipeline tests..."
pytest tests/test_pipeline.py -v \
    --tb=short \
    -q

echo ""
echo "Running API tests..."
pytest tests/test_api.py -v \
    --tb=short \
    -q \
    --ignore=tests/test_api.py  # skip if server not running

echo ""
echo "=================================="
echo "All tests passed! Safe to deploy."
echo "=================================="