    #!/usr/bin/env bash
    set -euo pipefail

    # SadTalker: script oficial (si cambia, consulta el repo)
    if [ -d "SadTalker" ]; then
      cd SadTalker
      if [ -f "scripts/download_models.sh" ]; then
        bash scripts/download_models.sh || true
      fi
      cd ..
    fi

    # Wav2Lip: descargar checkpoint GAN estable
    if [ -d "Wav2Lip" ]; then
      cd Wav2Lip
      mkdir -p checkpoints
      python - <<'PY'
import os, urllib.request
url = 'https://github.com/Rudrabha/Wav2Lip/releases/download/v0.1/wav2lip_gan.pth'
path = os.path.join('checkpoints','wav2lip_gan.pth')
os.makedirs('checkpoints', exist_ok=True)
try:
    urllib.request.urlretrieve(url, path)
    print('Descargado:', path)
except Exception as e:
    print('No se pudo descargar Wav2Lip checkpoint automáticamente:', e)
PY
      cd ..
    fi
