import os, shutil
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from pathlib import Path
from .utils import ensure_dir, unique_workdir
from .pipelines import (
    sadtalker_generate,
    wav2lip_refine,
    still_video_from_image,
    normalize_face_video,
)

app = FastAPI(title='Audio→Vídeo Avatar API', version='1.2.0')

DEVICE = os.getenv('DEVICE', 'cpu')  # Railway CPU por defecto
PORT = int(os.getenv('PORT', '8000'))

@app.get('/')
def root():
    return {'ok': True, 'device': DEVICE}

@app.get('/health')
def health():
    return {'status': 'healthy'}

@app.post('/generate')
async def generate(
    # Opción A: imagen + audio  (foto fija o SadTalker)
    image: UploadFile | None = File(default=None),
    # Opción B: vídeo de cara + audio  (Veo 2 + lip-sync)
    face_video: UploadFile | None = File(default=None),
    audio: UploadFile = File(...),

    # Parámetros
    refine_lips: bool = Form(True),     # si hay face_video, se usa Wav2Lip igualmente
    fps: int = Form(25),
    prefer: str = Form('auto'),         # 'auto' | 'sadtalker' | 'w2l'  (si hay face_video se ignora)
):
    """
    Modos:
      - face_video + audio  -> lip-sync directo con Wav2Lip (dinámico, sin --static).
      - image + audio       -> SadTalker (auto) o Wav2Lip estático según 'prefer'.
    """
    work = unique_workdir()
    try:
        # Guarda audio
        au_path = work / 'audio_in'
        with au_path.open('wb') as f:
            shutil.copyfileobj(audio.file, f)

        out_dir = work / 'out'
        ensure_dir(out_dir)

        # ---- B) Si nos pasan vídeo de cara -> lip-sync directo ----
        if face_video is not None:
            fv_path = work / 'face_input'
            with fv_path.open('wb') as f:
                shutil.copyfileobj(face_video.file, f)

            # Normaliza el vídeo (H.264 yuv420p, fps fijo) para evitar errores
            face_norm = normalize_face_video(fv_path, out_dir, fps=fps)

            final_path = work / 'final.mp4'
            # lip-sync dinámico (sin --static)
            final_video = wav2lip_refine(face_norm, au_path, final_path, device=DEVICE, static_mode=False)
            return FileResponse(path=str(final_video), media_type='video/mp4', filename='avatar.mp4')

        # ---- A) Si no hay vídeo, usamos imagen + audio ----
        if image is None:
            return JSONResponse({'ok': False, 'error': 'Debes subir image o face_video'}, status_code=400)

        img_path = work / 'image.jpg'
        with img_path.open('wb') as f:
            shutil.copyfileobj(image.file, f)

        base_video = None
        if prefer == 'w2l':
            # Solo Wav2Lip con imagen fija
            base_video = still_video_from_image(img_path, au_path, out_dir, fps=fps)
        else:
            # Intentar SadTalker y, si falla y prefer no obliga, caer a Wav2Lip estático
            try:
                base_video = sadtalker_generate(img_path, au_path, out_dir, fps=fps, device=DEVICE)
            except Exception as e:
                if prefer == 'sadtalker':
                    raise
                base_video = still_video_from_image(img_path, au_path, out_dir, fps=fps)

        final_path = work / 'final.mp4'
        if refine_lips:
            final_video = wav2lip_refine(base_video, au_path, final_path, device=DEVICE, static_mode=(prefer!='w2l' and image is not None))
        else:
            shutil.copy2(base_video, final_path)
            final_video = final_path

        return FileResponse(path=str(final_video), media_type='video/mp4', filename='avatar.mp4')

    except Exception as e:
        return JSONResponse({'ok': False, 'error': str(e)}, status_code=500)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run('app.server:app', host='0.0.0.0', port=PORT, reload=False)
