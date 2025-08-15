import os, shutil
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from pathlib import Path
from .utils import ensure_dir, unique_workdir
from .pipelines import sadtalker_generate, wav2lip_refine

app = FastAPI(title='Audio→Vídeo Avatar API', version='1.0.0')

DEVICE = os.getenv('DEVICE', 'cpu')  # for Railway CPU
PORT = int(os.getenv('PORT', '8000'))

@app.get('/')
def root():
    return {'ok': True, 'device': DEVICE}

@app.get('/health')
def health():
    return {'status': 'healthy'}

@app.post('/generate')
async def generate(
    image: UploadFile = File(...),
    audio: UploadFile = File(...),
    refine_lips: bool = Form(True),
    fps: int = Form(25)
):
    work = unique_workdir()
    try:
        img_path = work / 'image.jpg'
        au_path = work / 'audio_in'
        with img_path.open('wb') as f:
            shutil.copyfileobj(image.file, f)
        with au_path.open('wb') as f:
            shutil.copyfileobj(audio.file, f)

        out_dir = work / 'out'
        ensure_dir(out_dir)

        base_video = sadtalker_generate(img_path, au_path, out_dir, fps=fps, device=DEVICE)

        final_path = work / 'final.mp4'
        if refine_lips:
            final_video = wav2lip_refine(base_video, au_path, final_path, device=DEVICE)
        else:
            shutil.copy2(base_video, final_path)
            final_video = final_path

        return FileResponse(path=str(final_video), media_type='video/mp4', filename='avatar.mp4')
    except Exception as e:
        return JSONResponse({'ok': False, 'error': str(e)}, status_code=500)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run('app.server:app', host='0.0.0.0', port=PORT, reload=False)
