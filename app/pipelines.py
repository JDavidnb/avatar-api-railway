def normalize_image_to_png(image: Path, out_dir: Path) -> Path:
    """
    Re-codifica cualquier imagen de entrada (jpg/png/webp/heic/lo que sea) a PNG.
    Esto evita el error 'No JPEG data found in image'.
    """
    ensure_dir(out_dir)
    norm_img = out_dir / "img.png"
    # -hide_banner/-loglevel error para no spamear logs
    run([
        "ffmpeg","-y","-hide_banner","-loglevel","error",
        "-i", str(image),
        "-frames:v","1",
        str(norm_img)
    ])
    return norm_img

def still_video_from_image(image: Path, audio: Path, out_dir: Path, fps: int = 25) -> Path:
    """
    Crea un vídeo estático a partir de una imagen con la misma duración que el audio.
    Sirve como entrada para Wav2Lip cuando SadTalker falla/no está disponible.
    """
    ensure_dir(out_dir)
    # 1) Normaliza la imagen a PNG válido
    norm_img = normalize_image_to_png(image, out_dir)

    # 2) Calcula duración real del audio
    import json, subprocess
    res = subprocess.run(
        ["ffprobe","-v","error","-show_entries","format=duration","-of","json", str(audio)],
        capture_output=True, text=True, check=True
    )
    dur = float(json.loads(res.stdout)["format"]["duration"])

    # 3) Genera el vídeo estático (sin audio) a ese FPS/duración
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
