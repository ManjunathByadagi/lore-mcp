# Multi-stage build for production
FROM python:3.11-slim as builder

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install UV
RUN pip install uv

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml ./
COPY src/ ./src/

# Build wheel
RUN uv pip install build && python -m build

# Production stage
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    postgresql-client \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r lore && useradd -r -g lore lore

# Set working directory
WORKDIR /app

# Install UV
RUN pip install uv

# Copy built wheel from builder stage
COPY --from=builder /app/dist/*.whl ./

# Install the package
RUN uv pip install --system *.whl

# Create data directories
RUN mkdir -p /data/lore && \
    mkdir -p /data/logs && \
    chown -R lore:lore /data

# Copy entrypoint script
COPY docker/entrypoint.sh ./entrypoint.sh
RUN chmod +x entrypoint.sh

# Switch to non-root user
USER lore

# Set environment variables
ENV PYTHONPATH=/app
ENV KNOWLEDGE_DATA_DIR=/data/lore
ENV KNOWLEDGE_LOG_LEVEL=INFO

# Expose port (if running HTTP mode)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import lore; print('lore-mcp is healthy')" || exit 1

# Default command
ENTRYPOINT ["./entrypoint.sh"]
CMD ["lore-mcp"]