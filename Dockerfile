FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config.yaml .
COPY src/ src/
COPY api/ api/
COPY models/ models/

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]