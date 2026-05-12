# ============================================
# BharatRAG Dockerfile
# ============================================
# Every line is a LAYER in the image.
# Docker caches layers — unchanged layers
# don't rebuild. Order matters:
# Put rarely-changing things first (dependencies)
# Put frequently-changing things last (your code)
# ============================================

# Base image — Python 3.11 slim
# slim = smaller size, no unnecessary packages
# Java equivalent: FROM openjdk:17-slim
FROM python:3.11-slim

# Set working directory inside container
# All subsequent commands run from here
# Java equivalent: setting working dir in JAR
WORKDIR /app

# ── Install system dependencies ────────────
# These rarely change → good to cache early
# libgomp1 = required by sentence-transformers
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*
# rm -rf /var/lib/apt/lists/ = clean up apt cache
# Keeps image size smaller

# ── Install Python dependencies ────────────
# Copy requirements BEFORE copying code
# If requirements.txt unchanged → this layer cached
# Changing your code won't re-install packages
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt
# --no-cache-dir = don't cache pip downloads
# Keeps image smaller

# ── Copy application code ──────────────────
# Copied AFTER requirements so code changes
# don't invalidate the pip install cache layer
COPY api.py .
COPY src/ ./src/
COPY config.py .

# ── Environment variables ──────────────────
# These are DEFAULTS — override at runtime
# Never hardcode actual API keys here
ENV ANTHROPIC_API_KEY=""
ENV PINECONE_API_KEY=""
ENV PINECONE_INDEX="bharatrag"
ENV LANGSMITH_TRACING="false"
ENV LANGSMITH_API_KEY=""
ENV LANGSMITH_PROJECT="bharatrag"

# ── Expose port ────────────────────────────
# Tells Docker this container uses port 8000
# Doesn't actually open the port — just documents it
# Java equivalent: server.port=8000
EXPOSE 8000

# ── Health check ───────────────────────────
# Docker periodically calls this to check
# if the container is healthy
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ── Start command ──────────────────────────
# What runs when container starts
# Java equivalent: java -jar app.jar
CMD ["uvicorn", "api:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2"]
# workers=2 = handle 2 requests simultaneously
# Remove --reload (not needed in production)