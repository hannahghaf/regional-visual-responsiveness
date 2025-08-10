FROM python:3.10-slim
# py3.10 to be compatible w allensdk

# install system build tools & dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    procps \
    gcc \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# install python packages
RUN pip install --upgrade pip && \
    pip install allensdk pandas

WORKDIR /data
COPY bin/ /bin/
ENV DATA_DIR=/data
ENV PYTHONUNBUFFERED=1