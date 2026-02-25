FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements/ /tmp/requirements/
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /tmp/requirements/apps/weapon_monitor.txt

COPY models/weapon_detection/best_model.pth /app/models/best_model.pth
COPY src/ /app/src/
COPY apps/weapon_monitor/ /app/apps/weapon_monitor/

ENV PYTHONPATH="/app/src:$PYTHONPATH"
ENV WEAPON_MODEL_PATH=/app/models/best_model.pth

WORKDIR /app/apps/weapon_monitor

EXPOSE 5001

CMD ["python", "app.py"]
