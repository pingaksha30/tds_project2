FROM python:3.9-slim

WORKDIR /app
COPY . /app

# Install deps efficiently + avoid cache bloat
RUN pip install --no-cache-dir -r requirements.txt

# Preload minimal Whisper model separately to avoid RAM spike
ENV PYTHONUNBUFFERED=1
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
