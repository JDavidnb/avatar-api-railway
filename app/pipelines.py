import os
from pathlib import Path
from .utils import run, ensure_dir, ffmpeg_normalize_audio

ROOT = Path('/workspace')
SADTALKER = ROOT / 'SadTalker'
WAV2LIP = ROOT / 'Wav2Lip'

def sadtalker_generate(image: Path, audio: Path, out_dir: Path, fps: int = 25, device: str = 'cpu') -> Path:
    """Genera vídeo base desde imagen+audio con SadTalker."""
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

def still_video_from_image(image: Path, audio: Path, out_dir: Path, fps: int = 25) -> Path:
    """
    Crea un vídeo estático a partir de una imagen con la misma duración que el audio.
    Sirve como entrada para Wav2Lip cuando SadTalker falla/no está disponible.
    """
    ensure_dir(out_dir)
    # Duración del audio con ffprobe vía ffmpeg (más robusto que cargarlo en Python)
    # Generamos vídeo sin audio y con el fps deseado.
    out_video = out_dir / 'still.mp4'
    # Duración: usamos -stream_loop 1 + -t calculada por ffprobe, pero más simple:
    # duplicamos frames hasta que acabe Wav2Lip (que usa el audio aparte). Tomamos 300s tope.
    # Para mayor precisión, medimos duración real:
    import json, subprocess
    cmd = [
        'ffprobe','-v','error','-show_entries','format=duration','-of','json', str(audio)
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    dur = float(json.loads(res.stdout)['format']['duration'])
    # Generar vídeo estático
    run([
        'ffmpeg','-y',
        '-loop','1','-i',str(image),
        '-t',f'{dur:.3f}',
        '-vf',f'fps={fps},format=yuv420p',
        '-pix_fmt','yuv420p',
        str(out_video)
    ])
    return out_video

def wav2lip_refine(face_video: Path, audio: Path, out_path: Path, device: str = 'cpu') -> Path:
    """Refina sincronización labial con Wav2Lip (modo robusto en CPU)."""
    tmp_dir = out_path.parent
    norm_audio = tmp_dir / 'audio_16k_ref.wav'
    ffmpeg_normalize_audio(audio, norm_audio, sr=16000)

    ckpt_gan = WAV2LIP / 'checkpoints' / 'wav2lip_gan.pth'
    ckpt_base = WAV2LIP / 'checkpoints' / 'wav2lip.pth'
    ckpt = ckpt_gan if ckpt_gan.exists() else ckpt_base  # usa base si no hay GAN
    if not ckpt.exists():
        raise RuntimeError('Faltan pesos de Wav2Lip (ni wav2lip_gan.pth ni wav2lip.pth)')

    # Forzar modo "static", bajar resolución para que detecte mejor y dar margen alrededor de la cara
    env = os.environ.copy()
    if device == 'cpu':
        env['CUDA_VISIBLE_DEVICES'] = ''

    cmd = [
        'python3', 'inference.py',
        '--checkpoint_path', str(ckpt),
        '--face', str(face_video),
        '--audio', str(norm_audio),
        '--outfile', str(out_path),
        '--static', 'True',
        '--resize_factor', '2',
        '--pads', '0', '15', '0', '0',
        '--nosmooth'
    ]

    run(cmd, cwd=WAV2LIP)
    return out_path

