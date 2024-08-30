import numpy as np
from flask import Flask
from flask_cors import CORS
from fastapi import FastAPI
from typing import Any, Self
from pydantic import BaseModel, model_validator
from decimal import Decimal
from typing import List, Dict, Any
import requests
from starlette.middleware.cors import CORSMiddleware

from whisper import whisper
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


whisper_class = whisper()

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

class AudioProperties(BaseModel):
    chunk_index: int
    content: list[float | None]
    duration: float


    # @model_validator(mode="after")
    # def validate_content(self) -> Self:
    #     print(self.content)
    #     if not all(isinstance(n, float) for n in self.content):
    #         raise ValueError("Content must contain float data")
    #     return self

class MetadataProperties(BaseModel):
    device: str
    language: str

class Audio_data_request(BaseModel):
    audio: AudioProperties
    metadata: MetadataProperties
    is_final_chunk: bool
    timestamp: str
    user_id: str

class POST_DATE():
    def __init__(self):
        self.audio_chank = []
        self.audio_data = []
        self.content = []
        self.is_final_chunk = False
        self.user_id = ""
        self.timestamp = ""
        self.chunk_index = 1
        self.duration_chunk = 0
        self.duration_audio = 0
        self.device = "web"
        self.languages = "ru"

    def summiraze_audio(self):
        """
        Данная функция предназначенна для суммирования аудио чанка получаемого из FastAPI, подготовка данного аудио чанка для отправки на распознавание в Whisper
        :return: NaN
        """
        self.audio_chank = np.float32(np.frombuffer(self.content, np.int16)) # audio_chank хранит кусок данных которы отправиться на транскрипцию
        self.audio_data.append(self.audio_chank) # Сумируем audio_chank, что бы получить исходное аудио в переменной audio_data из кусочков с FastAPI

    def unpacking_data(self, data : Audio_data_request):
        self.user_id = data.user_id
        self.timestamp = data.timestamp

        self.chunk_index=data.audio.chunk_index
        self.content = data.audio.content
        self. duration_chunk = data.audio.duration

        self.device = data.metadata.device
        self.languages = data.metadata.language
        self.is_final_chunk = data.is_final_chunk



    def transcribe_audio(self):
        # Вызов whisper
        text = whisper.transcribe(data=self.audio_chank if self.is_final_chunk is False else self.audio_data)
        return text


@app.post(path="/proccess_audio")
async def proccess_audio(request_data: Audio_data_request):
    print(type(request_data.audio.content[0]))
    print(request_data.audio.content)
    audio_data = POST_DATE()
    audio_data.unpacking_data(request_data)
    print("succesfull")
    # audio_data.summiraze_audio()
    # text = audio_data.transcribe_audio()
    return request_data.metadata

app.mount('', StaticFiles(directory='./', html=True), name='static')


import uvicorn
uvicorn.run(app, host="localhost", port=8898)