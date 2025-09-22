# TODO: Considerar usar imagen Alpine para reducir tamaño
FROM python:3.11-slim

# Instalar dependencias del sistema para OpenCV y librerías de imagen
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgcc-s1 \
    libgl1-mesa-dri \
    libglib2.0-0 \
    libgstreamer1.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Establecer directorio de trabajo
WORKDIR /app

# Copiar requirements y instalar dependencias Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código de la aplicación
COPY . .

# Crear directorio para uploads
RUN mkdir -p uploads

# TODO: Crear usuario no-root para mayor seguridad
# RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
# USER appuser

# Exponer puerto 5000
EXPOSE 5000

# Variables de entorno
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

# Comando para ejecutar la aplicación
# TODO: Usar gunicorn para producción
CMD ["python", "app.py"]