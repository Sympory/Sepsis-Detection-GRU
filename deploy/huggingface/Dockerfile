# SepsisAI — Hugging Face Spaces (Docker) için herkese açık demo
FROM python:3.10-slim

# Kayıt saatleri Türkiye saatiyle tutulur
RUN apt-get update && apt-get install -y --no-install-recommends tzdata && rm -rf /var/lib/apt/lists/*
ENV TZ=Europe/Istanbul

# Hugging Face Spaces uygulamayı 1000 numaralı kullanıcıyla çalıştırır
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .
USER user

ENV PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    SEPSIS_PUBLIC_DEMO=1 \
    SEPSIS_HOST=0.0.0.0 \
    SEPSIS_PORT=7860 \
    SEPSIS_DB_PATH=/tmp/patients.db

EXPOSE 7860
CMD ["python", "space_start.py"]
