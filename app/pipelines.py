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


def wav2lip_refine(face_video: Path, audio: Path, out_path: Path, device: str = 'cpu') -> Path:
    """Refina sincronización labial con Wav2Lip."""
    tmp_dir = out_path.parent
    norm_audio = tmp_dir / 'audio_16k_ref.wav'
    ffmpeg_normalize_audio(audio, norm_audio, sr=16000)

    ckpt = WAV2LIP / 'checkpoints' / 'wav2lip_gan.pth'
    if not ckpt.exists():
        raise RuntimeError('Falta checkpoints de Wav2Lip (wav2lip_gan.pth)')

    if device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

    cmd = [
        'python3', 'inference.py',
        '--checkpoint_path', str(ckpt),
        '--face', str(face_video),
        '--audio', str(norm_audio),
        '--outfile', str(out_path)
    ]
    run(cmd, cwd=WAV2LIP)
    return out_path
