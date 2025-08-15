FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive         PYTHONUNBUFFERED=1         PIP_NO_CACHE_DIR=1         TORCH_HOME=/workspace/.cache/torch         DEVICE=cpu

# system deps
RUN apt-get update && apt-get install -y --no-install-recommends         git ffmpeg libgl1 libglib2.0-0 build-essential         && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# clone repos
RUN git clone --depth=1 https://github.com/OpenTalker/SadTalker.git      && git clone --depth=1 https://github.com/Rudrabha/Wav2Lip.git

# python deps (pines estables para Wav2Lip en CPU)
RUN python -m pip install --upgrade pip \
 && python -m pip install \
      numpy==1.26.4 \
      numba==0.58.1 \
      llvmlite==0.41.1 \
      opencv-python==4.8.1.78 \
      moviepy==1.0.3 \
      librosa==0.10.1 \
      scipy==1.11.4 \
      tqdm==4.66.4 \
      pillow==10.2.0 \
      ffmpeg-python==0.2.0 \
      typing-extensions==4.12.2 \
 && python -m pip install -r Wav2Lip/requirements.txt --no-deps || true \
 && python -m pip install -r SadTalker/requirements.txt --no-deps || true \
 && python -m pip install fastapi uvicorn[standard] python-multipart pydantic

# PyTorch CPU
RUN python -m pip install --index-url https://download.pytorch.org/whl/cpu \
      torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1

# download models at build time so they are baked into the image
COPY scripts/download_models.sh /workspace/scripts/download_models.sh
RUN bash /workspace/scripts/download_models.sh || true

# app code
COPY requirements.txt /workspace/requirements.txt
RUN python -m pip install -r /workspace/requirements.txt || true
COPY app /workspace/app

EXPOSE 8000
# Use $PORT if provided by the platform
CMD bash -lc 'uvicorn app.server:app --host 0.0.0.0 --port ${PORT:-8000}'
