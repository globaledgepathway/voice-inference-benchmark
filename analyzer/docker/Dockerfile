# Benchmark client only (the model servers use the official vLLM / Kokoro images).
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENTRYPOINT ["python", "-m", "via.cli"]
