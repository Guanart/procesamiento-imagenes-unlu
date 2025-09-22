# Usar imagen base de Python 3.9 slim para reducir tamaño
FROM python:3.9-slim

# Establecer directorio de trabajo en el contenedor
WORKDIR /app

# Instalar dependencias del sistema necesarias para Pillow
RUN apt-get update && apt-get install -y \
    gcc \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    libfreetype6-dev \
    liblcms2-dev \
    libopenjp2-7-dev \
    libtiff5-dev \
    libwebp-dev \
    && rm -rf /var/lib/apt/lists/*

# Copiar archivo de dependencias primero para aprovechar cache de Docker
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar código de la aplicación al contenedor
COPY . .

# Crear directorio de uploads
RUN mkdir -p uploads

# Exponer puerto 5000
EXPOSE 5000

# Configurar variables de entorno para Flask
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Crear usuario no privilegiado para ejecutar la aplicación
RUN adduser --disabled-password --gecos '' --shell /bin/bash user && \
    chown -R user:user /app
USER user

# Comando para ejecutar la aplicación
CMD ["python", "app.py"]