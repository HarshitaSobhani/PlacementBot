FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ api/
COPY frontend/ frontend/
COPY model/best_model.pkl model/preprocessor.pkl model/metadata.json model/background_transformed.npy model/

# HF Spaces (Docker SDK) routes traffic to port 7860 by default.
EXPOSE 7860
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
