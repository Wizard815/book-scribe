# Build context is the repo root (see docker-compose.yml) so the vendored mem0 is available.
FROM python:3.12-slim
WORKDIR /app
COPY agent/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY vendor/mem0 /tmp/mem0
RUN pip install --no-cache-dir --no-deps /tmp/mem0 && rm -rf /tmp/mem0
COPY agent/ .
ENV MEM0_TELEMETRY=False
EXPOSE 8000
CMD ["uvicorn", "main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
