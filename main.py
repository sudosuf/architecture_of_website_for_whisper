import numpy as np
import soundfile as sf
import whisper
from fastapi import FastAPI, File, UploadFile
from io import BytesIO
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles


model = whisper.load_model('tiny')



app = FastAPI()
origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ProcessAudioPost(BaseModel):
    pass

@app.post(path="/process_audio")
async def process_audio(file: UploadFile = File(...), is_final_chunk: bool = False):
    audio_bytes = await file.read()
    audio_buffer = BytesIO(audio_bytes)
    audio_data, samplerate = sf.read(audio_buffer)

    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)
    audio_data = audio_data.astype(np.float32)

    result = model.transcribe(audio_data)

    return {"transcription": result['text']}

app.mount('', StaticFiles(directory='./', html=True), name='static')