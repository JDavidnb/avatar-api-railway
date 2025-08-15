FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive         PYTHONUNBUFFERED=1         PIP_NO_CACHE_DIR=1         TORCH_HOME=/workspace/.cache/torch         DEVICE=cpu

# system deps
RUN apt-get update && apt-get install -y --no-install-recommends         git ffmpeg libgl1 libglib2.0-0 build-essential         && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# clone repos
RUN git clone --depth=1 https://github.com/OpenTalker/SadTalker.git      && git clone --depth=1 https://github.com/Rudrabha/Wav2Lip.git

# python deps
RUN python -m pip install --upgrade pip      && python -m pip install -r Wav2Lip/requirements.txt || true      && python -m pip install -r SadTalker/requirements.txt || true      && python -m pip install fastapi uvicorn[standard] opencv-python-headless             ffmpeg-python moviepy librosa numpy scipy tqdm huggingface_hub gdown      && python -m pip install --index-url https://download.pytorch.org/whl/cpu             torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1

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
