import os, json, subprocess
from pathlib import Path
from .utils import run, ensure_dir, ffmpeg_normalize_audio

ROOT = Path('/workspace')
SADTALKER = ROOT / 'SadTalker'
WAV2LIP = ROOT / 'Wav2Lip'

# ----------------- SadTalker (por si se usa en auto) -----------------
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

# ----------------- Imagen -> PNG y vídeo estático -----------------
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
    # Duración real del audio
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

# ----------------- Normalizar vídeo de cara (Veo 2) -----------------
def normalize_face_video(face_video: Path, out_dir: Path, fps: int = 25) -> Path:
    """
    Re-encode a H.264 yuv420p y fps fijo. Evita errores de pixel format / timestamps.
    """
    ensure_dir(out_dir)
    out_face = out_dir / "face_norm.mp4"
    run([
        "ffmpeg","-y","-hide_banner","-loglevel","error",
        "-i", str(face_video),
        "-vf", f"fps={fps},scale=iw:ih,format=yuv420p",
        "-c:v","libx264","-preset","veryfast","-crf","18",
        "-an",  # sin audio
        str(out_face)
    ])
    return out_face

# ----------------- Lip-sync con Wav2Lip -----------------
def wav2lip_refine(face_video: Path, audio: Path, out_path: Path, device: str = 'cpu', static_mode: bool = False) -> Path:
    """
    Si static_mode=True: foto fija (--static True).
    Si static_mode=False: vídeo real (sin --static).
    """
    tmp_dir = out_path.parent
    norm_audio = tmp_dir / 'audio_16k_ref.wav'
    ffmpeg_normalize_audio(audio, norm_audio, sr=16000)

    # Usa GAN si existe; si no, el modelo base
    ckpt_gan = WAV2LIP / 'checkpoints' / 'wav2lip_gan.pth'
    ckpt_base = WAV2LIP / 'checkpoints' / 'wav2lip.pth'
    ckpt = ckpt_gan if ckpt_gan.exists() else ckpt_base
    if not ckpt.exists():
        raise RuntimeError('Faltan pesos de Wav2Lip (ni wav2lip_gan.pth ni wav2lip.pth)')

    if device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

    cmd = [
        'python3', 'inference.py',
        '--checkpoint_path', str(ckpt),
        '--face', str(face_video),
        '--audio', str(norm_audio),
        '--outfile', str(out_path),
        '--resize_factor', '2',
        '--pads', '0', '15', '0', '0',
        '--nosmooth'
    ]
    if static_mode:
        # Algunas versiones requieren valor explícito
        cmd.extend(['--static','True'])

    run(cmd, cwd=WAV2LIP)
    return out_path
