# Build context is the repo root (see docker-compose.yml) so the vendored mem0 and graphify are available.
FROM python:3.12-slim
WORKDIR /app
COPY agent/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY vendor/mem0 /tmp/mem0
COPY vendor/graphify /tmp/graphify
RUN pip install --no-cache-dir --no-deps /tmp/mem0 && pip install --no-cache-dir /tmp/graphify && rm -rf /tmp/mem0 /tmp/graphify
COPY agent/ .
ENV MEM0_TELEMETRY=False
EXPOSE 8000
CMD ["uvicorn", "main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
