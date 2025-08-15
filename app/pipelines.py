import os, json, subprocess, urllib.request
from pathlib import Path
from .utils import run, ensure_dir, ffmpeg_normalize_audio

ROOT = Path('/workspace')
SADTALKER = ROOT / 'SadTalker'
WAV2LIP = ROOT / 'Wav2Lip'

# --------- util descarga robusta ----------
def _download(url: str, dest: Path) -> bool:
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=600) as r, open(dest, "wb") as f:
            f.write(r.read())
        return True
    except Exception as e:
        print(f"[download] fallo {url}: {e}")
        return False

# --------- SadTalker (si lo usas) ----------
def sadtalker_generate(image: Path, audio: Path, out_dir: Path, fps: int = 25, device: str = 'cpu') -> Path:
    ensure_dir(out_dir)
    norm_audio = out_dir / 'audio_16k.wav'
    ffmpeg_normalize_audio(audio, norm_audio, sr=16000)

    cmd = [
        'python3', 'inference.py',
        '--driven_audio', str(norm_audio),
        '--source_image', str(image),
        '--result_dir', str(out_dir),
        '--fps', str(fps),
        '--still'
    ]
    if device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    run(cmd, cwd=SADTALKER)

    vids = sorted(out_dir.glob('*.mp4'), key=lambda p: p.stat().st_mtime, reverse=True)
    if not vids:
        raise RuntimeError('SadTalker no produjo vídeo')
    return vids[0]

# --------- Normalizar imagen y vídeo estático ----------
def normalize_image_to_png(image: Path, out_dir: Path) -> Path:
    ensure_dir(out_dir)
    norm_img = out_dir / "img.png"
    run([
        "ffmpeg","-y","-hide_banner","-loglevel","error",
        "-i", str(image), "-frames:v","1", str(norm_img)
    ])
    return norm_img

def still_video_from_image(image: Path, audio: Path, out_dir: Path, fps: int = 25) -> Path:
    ensure_dir(out_dir)
    norm_img = normalize_image_to_png(image, out_dir)
    res = subprocess.run(
        ["ffprobe","-v","error","-show_entries","format=duration","-of","json", str(audio)],
        capture_output=True, text=True, check=True
    )
    dur = float(json.loads(res.stdout)["format"]["duration"])
    out_video = out_dir / "still.mp4"
    run([
        "ffmpeg","-y","-hide_banner","-loglevel","error",
        "-loop","1","-i", str(norm_img), "-t", f"{dur:.3f}",
        "-vf", f"fps={fps},format=yuv420p", "-pix_fmt","yuv420p",
        str(out_video)
    ])
    return out_video

# --------- Pesos necesarios (multi-mirror) ----------
def ensure_face_detector() -> Path:
    """Asegura S3FD para Wav2Lip."""
    ckpt = WAV2LIP / "face_detection" / "detection" / "sfd" / "s3fd.pth"
    if ckpt.exists(): return ckpt
    mirrors = [
        # mirror oficial del autor del detector
        "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth",  # :contentReference[oaicite:1]{index=1}
        # mirrors en HuggingFace
        "https://huggingface.co/camenduru/facexlib/resolve/main/s3fd-619a316812.pth",  # :contentReference[oaicite:2]{index=2}
        "https://huggingface.co/wsj1995/sadTalker/resolve/main/s3fd-619a316812.pth",   # :contentReference[oaicite:3]{index=3}
    ]
    tmp = ckpt.parent / "s3fd-619a316812.pth"
    for u in mirrors:
        if _download(u, tmp):
            tmp.replace(ckpt)
            return ckpt
    raise RuntimeError("No pude descargar s3fd.pth (S3FD)")

def ensure_wav2lip_weights() -> Path:
    """
    Intenta primero el GAN (mejor boca).
    Si falla, baja el modelo base 'wav2lip.pth'.
    Devuelve la ruta que exista.
    """
    ckpt_gan = WAV2LIP / "checkpoints" / "wav2lip_gan.pth"
    if ckpt_gan.exists(): return ckpt_gan
    ckpt_base = WAV2LIP / "checkpoints" / "wav2lip.pth"
    if ckpt_base.exists(): return ckpt_base

    mirrors_gan = [
        "https://huggingface.co/Nekochu/Wav2Lip/resolve/main/wav2lip_gan.pth",  # :contentReference[oaicite:4]{index=4}
        "https://huggingface.co/Non-playing-Character/Wave2lip/resolve/main/wav2lip_gan.pth",  # :contentReference[oaicite:5]{index=5}
    ]
    for u in mirrors_gan:
        if _download(u, ckpt_gan):
            return ckpt_gan

    mirrors_base = [
        "https://huggingface.co/numz/wav2lip_studio/resolve/main/Wav2lip/wav2lip.pth",  # :contentReference[oaicite:6]{index=6}
    ]
    for u in mirrors_base:
        if _download(u, ckpt_base):
            return ckpt_base

    rais
