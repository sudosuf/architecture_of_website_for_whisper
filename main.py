import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, UploadFile, Request
from io import BytesIO
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torch
import time
import logging
import traceback
################################################# ЛОГИРОВАНИЕ ##########################################################

class HostLoggerAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        client_host = kwargs.get("extra", {}).get("client_host", "unknown")
        return f"[{client_host}] {msg}", kwargs


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(client_host)s] %(message)s",
    handlers=[
        logging.FileHandler("server_logs.log"),
        logging.StreamHandler()
    ]
)

logger = HostLoggerAdapter(logging.getLogger(__name__), {})

########################################################################################################################

# model = whisper.load_model('tiny')
##################################### ИНИЦИАЛИЗАЦИЯ МОДЕЛИ WHISPER #####################################################
dict = {str: np.float32}
print("Cтатус видеокарты: ", torch.cuda.is_available())
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    batch_size=16,
    torch_dtype=torch_dtype,
    device=device,
)
########################################################################################################################
############################################ ИНИЦИАЛИЗАЦИЯ FAST API ####################################################
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
########################################################################################################################
class ProcessAudioPost(BaseModel):
    pass

def concatinate_audio_chank(host, audio_data):
    if str(host) in dict:
        audio_from_dict = dict[f"{str(host)}"]
        dict[f"{str(host)}"] = np.concatenate((audio_from_dict, audio_data))
    else:
        dict[f"{str(host)}"] = audio_data

@app.post(path="/process_audio")
async def process_audio(request: Request, file: UploadFile = File(...), is_final_chunk: bool = False):
    try:
        time_query = time.strftime('%Y%m%d_%H%M%S')  # Freeztime время отправления файла на сервер
        host = request.client.host  # охранение ip  с которого поступил файл
        logger.info("Received a new audio chunk", extra={"client_host": host})
        audio_bytes = await file.read()  # Cчитывание файла с API


        audio_buffer = BytesIO(audio_bytes)  # Преобразуем файл в последовательность битов

        try:
            audio_data, samplerate = sf.read(audio_buffer)  # считываем последовательность бидов в массив аудиосигнала
        except Exception as e:
            logger.warning(f"Error in worked: uncorrected format file", extra={"client_host": host})
            return {"Error": "Uncorrected format file"}

        concatinate_audio_chank(host, audio_data)  # Обьединение чанка с предыдущими чанками с этого ip

        audio_for_asr = dict[f"{str(host)}"]  # Cчитываем объединеную последовательность из словаря

        if len(audio_for_asr.shape) > 1:
            audio_for_asr = np.mean(audio_for_asr, axis=1)  # Превращаем стерео в моно
            logger.warning("Getting a stereo audio", extra={"client_host": host})

        audio_data = audio_for_asr.astype(np.float32)  # Переводим массив аудио в формат float32

        result = pipe(audio_data, generate_kwargs={"language": "russian"})
        logger.info("Recognized audio chunk", extra={"client_host": host})

        if is_final_chunk:
            dict[f"{str(host)}"] = [0]
            sf.write((f"save-audio/{host}---{time_query}.wav".replace('.', '-', 3)).replace('_', '-', 1), audio_data,
                     16000)
            logger.info(f"Final transcription: {result['text']}", extra={"client_host": host})
            logger.info(
                f"Save audio to file -- save-audio/{host}---{time_query}.wav".replace('.', '-', 3).replace('_', '-', 1),
                extra={"client_host": host})

        return {"transcription": result['text']}
    except Exception as e:
        host = request.client.host
        logger.critical(f"Error in worked: {traceback.format_exc()}", extra={"client_host": host})
        return {"Error": "Unknown error"}


app.mount('', StaticFiles(directory='./', html=True), name='static')
