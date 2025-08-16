import os, json, subprocess
from pathlib import Path
from .utils import run, ensure_dir, ffmpeg_normalize_audio

ROOT = Path('/workspace')
SADTALKER = ROOT / 'SadTalker'
WAV2LIP = ROOT / 'Wav2Lip'

# ----------------- SadTalker (por si se usa) -----------------
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

# ----------------- Utiles -----------------
def media_duration(path: Path) -> float:
    res = subprocess.run(
        ["ffprobe","-v","error","-show_entries","format=duration","-of","json", str(path)],
        capture_output=True, text=True, check=True
    )
    return float(json.loads(res.stdout)["format"]["duration"])

def normalize_image_to_png(image: Path, out_dir: Path) -> Path:
    ensure_dir(out_dir)
    out_img = out_dir / "img.png"
    run(["ffmpeg","-y","-hide_banner","-loglevel","error","-i",str(image),"-frames:v","1",str(out_img)])
    return out_img

def still_video_from_image(image: Path, audio: Path, out_dir: Path, fps: int = 25) -> Path:
    ensure_dir(out_dir)
    norm_img = normalize_image_to_png(image, out_dir)
    dur = media_duration(audio)
    out_video = out_dir / "still.mp4"
    run([
        "ffmpeg","-y","-hide_banner","-loglevel","error",
        "-loop","1","-i", str(norm_img),
        "-t", f"{dur:.3f}",
        "-vf", f"fps={fps},format=yuv420p",
        "-pix_fmt","yuv420p",
        str(out_video)
    ])
    return out_video

def normalize_face_video(face_video: Path, out_dir: Path, fps: int = 25) -> Path:
    """Re-encode H.264 yuv420p y fps fijo para evitar problemas de formatos."""
    ensure_dir(out_dir)
    out_face = out_dir / "face_norm.mp4"
    run([
        "ffmpeg","-y","-hide_banner","-loglevel","error",
        "-i", str(face_video),
        "-vf", f"fps={fps},scale=iw:ih,format=yuv420p",
        "-c:v","libx264","-preset","veryfast","-crf","18",
        "-an",
        str(out_face)
    ])
    return out_face

def choose_wav2lip_ckpt() -> Path:
    """En CPU prefiero el modelo base si existe (más ligero); si no, uso GAN."""
    ckpt_base = WAV2LIP / 'checkpoints' / 'wav2lip.pth'
    ckpt_gan  = WAV2LIP / 'checkpoints' / 'wav2lip_gan.pth'
    if ckpt_base.exists():
        return ckpt_base
    if ckpt_gan.exists():
        return ckpt_gan
    raise RuntimeError('Faltan pesos de Wav2Lip (ni wav2lip.pth ni wav2lip_gan.pth)')

# ----------------- Wav2Lip: chunked -----------------
def _slice_video(src: Path, start: float, dur: float, out_path: Path, fps: int):
    run([
        "ffmpeg","-y","-hide_banner","-loglevel","error",
        "-ss", f"{start:.3f}", "-t", f"{dur:.3f}",
        "-i", str(src),
        "-vf", f"fps={fps},format=yuv420p",
        "-c:v","libx264","-preset","veryfast","-crf","18",
        "-an",
        str(out_path)
    ])

def _slice_audio(src: Path, start: float, dur: float, out_path: Path):
    run([
        "ffmpeg","-y","-hide_banner","-loglevel","error",
        "-ss", f"{start:.3f}", "-t", f"{dur:.3f}",
        "-i", str(src),
        "-vn","-ac","1","-ar","16000","-c:a","pcm_s16le",
        str(out_path)
    ])

def wav2lip_refine_chunked(face_video: Path, audio: Path, out_path: Path, device: str = 'cpu',
                           fps: int = 25, chunk_sec: int = 10, static_mode: bool = False) -> Path:
    """
    Divide el vídeo y el audio en segmentos de 'chunk_sec' segundos, aplica Wav2Lip por partes
    (batch pequeño y resize) y concatena.
    """
    ensure_dir(out_path.parent)
    ckpt = choose_wav2lip_ckpt()

    # límites
    total = media_duration(audio)
    n = max(1, int(total // chunk_sec) + (1 if (total % chunk_sec) > 0.25 else 0))

    seg_outs = []
    for i in range(n):
        start = i * chunk_sec
        dur = min(chunk_sec, total - start)
        seg_dir = out_path.parent / f"seg_{i:03d}"
        ensure_dir(seg_dir)

        v_seg = seg_dir / "v.mp4"
        a_seg = seg_dir / "a.wav"
        _slice_video(face_video, start, dur, v_seg, fps=fps)
        _slice_audio(audio, start, dur, a_seg)

        seg_out = seg_dir / "out.mp4"
        cmd = [
            'python3','inference.py',
            '--checkpoint_path', str(ckpt),
            '--face', str(v_seg),
            '--audio', str(a_seg),
            '--outfile', str(seg_out),
            '--resize_factor', '1' if device=='cuda' else '2',
            '--pads','0','15','0','0',
            '--wav2lip_batch_size','8',
            '--face_det_batch_size','1',
            '--nosmooth'
        ]
        if static_mode:
            cmd.extend(['--static','True'])
        if device == 'cpu':
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
        run(cmd, cwd=WAV2LIP)
        seg_outs.append(seg_out)

    # Concatena (reencode para evitar incompatibilidades de streams)
    concat_list = out_path.parent / "concat.txt"
    with concat_list.open("w") as f:
        for p in seg_outs:
            f.write(f"file '{p.as_posix()}'\n")

    run([
        "ffmpeg","-y","-hide_banner","-loglevel","error",
        "-f","concat","-safe","0","-i", str(concat_list),
        "-c:v","libx264","-preset","veryfast","-crf","18",
        "-c:a","aac","-b:a","192k","-movflags","+faststart",
        str(out_path)
    ])
    return out_path

def wav2lip_refine(face_video: Path, audio: Path, out_path: Path, device: str = 'cpu', static_mode: bool = False) -> Path:
    """
    Si el clip > 20 s, usa versión 'chunked' automáticamente.
    """
    dur = media_duration(audio)
    if dur > 20:
        # chunk de 10 s por defecto
        return wav2lip_refine_chunked(face_video, audio, out_path, device=device, chunk_sec=10, static_mode=static_mode)
    # clip corto: modo directo
    ckpt = choose_wav2lip_ckpt()
    tmp_dir = out_path.parent
    norm_audio = tmp_dir / 'audio_16k_ref.wav'
    ffmpeg_normalize_audio(audio, norm_audio, sr=16000)
    cmd = [
        'python3','inference.py',
        '--checkpoint_path', str(ckpt),
        '--face', str(face_video),
        '--audio', str(norm_audio),
        '--outfile', str(out_path),
        '--resize_factor','2',
        '--pads','0','15','0','0',
        '--wav2lip_batch_size','8',
        '--face_det_batch_size','1',
        '--nosmooth'
    ]
    if static_mode:
        cmd.extend(['--static','True'])
    if device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    run(cmd, cwd=WAV2LIP)
    return out_path
