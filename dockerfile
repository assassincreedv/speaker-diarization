FROM nvcr.io/nvidia/cuda:12.2.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.10 \
        python3-pip \
        python3.10-venv \
        ffmpeg \
        build-essential && \
    rm -rf /var/lib/apt/lists/* && \
    ln -s /usr/bin/python3.10 /usr/local/bin/python && \
    ln -s /usr/bin/pip3 /usr/local/bin/pip

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir gunicorn

COPY . .

# 这里把 EXPOSE 改为 8082
EXPOSE 8082

# 把 gunicorn 绑定端口改为 8082
CMD ["gunicorn", "--workers", "4", "--bind", "0.0.0.0:8082", "wsgi:app"]
