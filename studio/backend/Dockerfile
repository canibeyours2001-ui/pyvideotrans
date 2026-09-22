FROM python:3.11-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg fonts-noto-core && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY backend/requirements.txt backend/requirements.txt
RUN pip3 install --no-cache-dir -r backend/requirements.txt
RUN pip3 install --no-cache-dir nvidia-cublas-cu12==12.4.5.8 nvidia-cudnn-cu12==9.1.0.70
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.11/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib
COPY backend backend
ENV STUDIO_DATA=/data MODEL_CACHE=/models PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["uvicorn","backend.local:api","--host","0.0.0.0","--port","8000","--workers","1"]
