# ---- Build Stage ----
# Use a full Python image to build the wheel
FROM python:3.9 as builder

WORKDIR /app

# Copy only necessary files for building
COPY pyproject.toml README.md LICENSE /app/
COPY core /app/core
COPY meta_rl /app/meta_rl
COPY env_runner /app/env_runner
COPY adaptation /app/adaptation
COPY experience /app/experience

# Install build dependencies and build the wheel
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir build && \
    python -m build --wheel --outdir dist .

# ---- Final Stage ----
# Use a slim image for the final runtime
FROM python:3.9-slim

WORKDIR /app

# Copy the built wheel from the builder stage
COPY --from=builder /app/dist/*.whl /tmp/

# Install the wheel and runtime dependencies
RUN pip install --no-cache-dir /tmp/*.whl && \
    rm /tmp/*.whl

# Copy application scripts
COPY main.py /app/main.py
COPY test_integration.py /app/test_integration.py
COPY experiments /app/experiments
COPY tests /app/tests

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Health check (optional)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import torch; import gymnasium; print('OK')" || exit 1

# Default command shows help
CMD ["python", "main.py", "--help"]

