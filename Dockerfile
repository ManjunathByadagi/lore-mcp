FROM python:3.11-slim

WORKDIR /app

# Install the package from PyPI
RUN pip install --no-cache-dir lore-knowledge-mcp

# Create data directory
RUN mkdir -p /data/lore

# Environment
ENV DB_BACKEND=sqlite
ENV KNOWLEDGE_DATA_DIR=/data/lore
ENV LOG_LEVEL=INFO

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["lore-mcp", "--host", "0.0.0.0", "--port", "8000"]
