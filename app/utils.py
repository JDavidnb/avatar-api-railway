import os, subprocess, uuid, shutil
from pathlib import Path

def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def run(cmd: list[str], cwd: str | Path | None = None):
    print('[RUN]', ' '.join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)

def ffmpeg_normalize_audio(in_path: str | Path, out_path: str | Path, sr: int = 16000):
    run([
        'ffmpeg','-y','-i',str(in_path),'-vn','-ac','1','-ar',str(sr),'-c:a','pcm_s16le',str(out_path)
    ])

def unique_workdir(base: str = '/workspace/work') -> Path:
    d = ensure_dir(base) / str(uuid.uuid4())
    d.mkdir(parents=True, exist_ok=True)
    return d
