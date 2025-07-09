FROM python:3.10-slim AS builder

WORKDIR /app

# install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

# install python requirements first for caching
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt uvicorn[standard]

# copy source
COPY . .

FROM python:3.10-slim

WORKDIR /app

# copy dependencies from builder
COPY --from=builder /usr/local /usr/local
COPY --from=builder /app /app

# ensure directories exist
RUN mkdir -p /app/logs /app/models

EXPOSE 8000
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
