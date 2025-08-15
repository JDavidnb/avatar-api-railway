#!/usr/bin/env bash
set -euo pipefail

# 1) SadTalker (usa su propio script si existe)
if [ -d "SadTalker" ]; then
  cd SadTalker
  if [ -f "scripts/download_models.sh" ]; then
    bash scripts/download_models.sh || true
  fi
  cd ..
fi

# 2) Wav2Lip: pesos + detector de caras S3FD (multi-mirror)
if [ -d "Wav2Lip" ]; then
  cd Wav2Lip
  mkdir -p checkpoints face_detection/detection/sfd
  python - <<'PY'
import os, urllib.request

def dl(url, dest):
    try:
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        req = urllib.request.Request(url, headers={"User-Agent":"Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=600) as r, open(dest, "wb") as f:
            f.write(r.read())
        print("OK", url, "->", dest)
        return True
    except Exception as e:
        print("FAIL", url, e)
        return False

# GAN primero; si falla, modelo base
ckpt_gan = "checkpoints/wav2lip_gan.pth"
ckpt_base = "checkpoints/wav2lip.pth"

mirrors_gan = [
    "https://huggingface.co/Nekochu/Wav2Lip/resolve/main/wav2lip_gan.pth",
    "https://huggingface.co/Non-playing-Character/Wave2lip/resolve/main/wav2lip_gan.pth",
]
if not os.path.exists(ckpt_gan):
    for u in mirrors_gan:
        if dl(u, ckpt_gan):
            break

if (not os.path.exists(ckpt_gan)) and (not os.path.exists(ckpt_base)):
    dl("https://huggingface.co/numz/wav2lip_studio/resolve/main/Wav2lip/wav2lip.pth", ckpt_base)

# S3FD detector
s3fd = "face_detection/detection/sfd/s3fd.pth"
mirrors_s3fd = [
    "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth",
    "https://huggingface.co/camenduru/facexlib/resolve/main/s3fd-619a316812.pth",
    "https://huggingface.co/wsj1995/sadTalker/resolve/main/s3fd-619a316812.pth",
]
if not os.path.exists(s3fd):
    for u in mirrors_s3fd:
        if dl(u, s3fd):
            break
PY
  cd ..
fi
