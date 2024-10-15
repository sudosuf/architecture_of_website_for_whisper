import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, UploadFile, Request
from io import BytesIO
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from transformers import (AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, WhisperForConditionalGeneration,
                          WhisperProcessor, WhisperTokenizer, AutomaticSpeechRecognitionPipeline)
from peft import PeftModel, PeftConfig
import torch
import time
import logging
import traceback
import json
from pydub import AudioSegment
import threading as th
#Нужно сохранять в буфере по 2 элемента, а затем искать одинаковораспознаные элементы в текущем чанке и суммированном тексте: "Покажи объем добычи 15 или *8*", "или 8 марта".
# В результирующем тексте после каждого чанка, надо не выводить не по 1 слову, а по два (95 строка) и вырезать из массива тайминг начала 2-го слова с конца (105 строка)
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
#################################### ОБЯВЛЕНИЕ СЛОВАРЕЙ ################################################################

dict = {str: np.float32}
dict_ending = {str: np.float32}
dict_recognise_summary = {str: float}

##################################### ИНИЦИАЛИЗАЦИЯ МОДЕЛИ WHISPER #####################################################
print("Cтатус видеокарты: ", torch.cuda.is_available())
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
task = "automatic-speech-recognition"
model_id = "openai/whisper-small"
path_to_model = "Whisper/model"
path_to_adapter = "Whisper/adapter"
peft_config = PeftConfig.from_pretrained(path_to_adapter)
model = WhisperForConditionalGeneration.from_pretrained(
    model_id
)
#
#model = PeftModel.from_pretrained(model, model_id=path_to_adapter)
model.to(device)
tokenizer = WhisperTokenizer.from_pretrained(model_id, language="russian", task=task)
processor = WhisperProcessor.from_pretrained(model_id, language="russian", task=task)

pipe = AutomaticSpeechRecognitionPipeline(
    model=model,
    tokenizer=tokenizer,
    feature_extractor=processor.feature_extractor,
    batch_size=1,
    torch_dtype=torch_dtype,
    device=device,
    return_timestamps="word"
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


def summaryRecognize(text: str, host):
    text = text.split(" ")
    print("text, summaryRecognixe : ", text)
    for word in text[:-1]:
        if str(host) in dict_recognise_summary:
            dict_recognise_summary[f"{str(host)}"] = dict_recognise_summary[f"{str(host)}"] + " " + word
        else:
            dict_recognise_summary[f"{str(host)}"] = word
    print("dict_recognise_summary: ", dict_recognise_summary[f"{str(host)}"])
    return dict_recognise_summary[f"{str(host)}"]


def findLastTimestampsAndSave(data, audio_massive, host):
    audio_last_word = []
    #last_word = data['chunks']
    #print("last_word: ", last_word)
    start_time = 0
    end_time = -1
    for chunk in data['chunks']:
            start_time, end_time = chunk['timestamp']
    print("start_time:", start_time)
    print("end_time:", end_time)
    print(audio_last_word)
    dict_ending[f'{str(host)}'] = audio_massive[int(16000*float(start_time)):]

def add_audio_to_dict(host, audio_data): #secund:float):
    if str(host) in dict:
        audio_from_dict = dict[f"{str(host)}"]
        dict[f"{str(host)}"] = np.concatenate((audio_from_dict, audio_data)) # Сумирую аудио данные в переменую в словаре
        return np.concatenate((dict_ending[f"{str(host)}"], audio_data)) # Возвращаю объединение окончание предыдущего чанка и текущий чанк
        #dict_ending[f"{str(host)}"] = np.concatenate((audio_from_dict[-16000*secund:], audio_data))
    else:
        dict[f"{str(host)}"] = audio_data
        dict_ending[f"{str(host)}"] = audio_data
        return audio_data


def extract_audio_massive_from_bytes(audio_bytes, host):
    audio_buffer = BytesIO(audio_bytes)  # Преобразуем файл в последовательность битов

    try:
        audio_data, samplerate = sf.read(audio_buffer)  # считываем последовательность битов в массив аудиосигнала
        return audio_data, samplerate
    except Exception as e:
        logger.warning(f"Error in worked: uncorrected format file", extra={"client_host": host})
        return {"Error": "Uncorrected format file"}


def add_logger_information(request):
    time_query = time.strftime('%Y%m%d_%H%M%S')  # Freeztime время отправления файла на сервер
    host = request.client.host  # охранение ip  с которого поступил файл
    logger.info("Received a new audio chunk", extra={"client_host": host})
    return time_query


def prepaireAudio(host, audio_bytes):


    audio_data, samplerate = extract_audio_massive_from_bytes(audio_bytes, host)

    audio_data = add_audio_to_dict(host, audio_data)  # Обьединение чанка с предыдущими чанками с этого ip

    # audio_for_asr = dict_ending[f"{str(host)}"]  # Cчитываем объединеную последовательность из словаря

    if len(audio_data.shape) > 1:
        audio_for_asr = np.mean(audio_data, axis=1)  # Превращаем стерео в моно
        logger.warning("Getting a stereo audio", extra={"client_host": host})

    audio_data = audio_data.astype(np.float32)  # Переводим массив аудио в формат float32
    return audio_data

def saveAudioToFile(host, time_query, result, result_audio):
    dict[f"{str(host)}"] = [0]
    sf.write((f"save-audio/{host}---{time_query}.wav".replace('.', '-', 3)).replace('_', '-', 1),
             result_audio,
             16000)
    logger.info(f"Final transcription: {result['text']}", extra={"client_host": host})
    logger.info(
        f"Save audio to file -- save-audio/{host}---{time_query}.wav".replace('.', '-', 3).replace('_', '-', 1),
        extra={"client_host": host})


@app.post(path="/process_audio")
async def process_audio(request: Request, file: UploadFile = File(...), is_final_chunk: bool = False):
    audio_bytes = await file.read()  # Cчитывание файла с API
    time_start = time.time()
    time_query = add_logger_information(request)
    host = request.client.host  # охранение ip  с которого поступил файл

    if not is_final_chunk:
        try:

            audio_data = prepaireAudio(host, audio_bytes) #Подготовка файла к распознаванию по чанкам

            result = pipe(audio_data, generate_kwargs={"language": "russian"})

            pr = th.Thread(target=findLastTimestampsAndSave, args=(result, audio_data, host))
            pr.start()

            summury_text = summaryRecognize(result["text"], host)  # Сумируем распознвный текст в словарь

            print("result: ", result)

            logger.info("Recognized audio chunk", extra={"client_host": host})

            time_end = time.time()
            print("time for work: ", time_end - time_start)
            pr.join(timeout=10.00)
            return {"transcription": summury_text}
        except Exception as e:
            host = request.client.host
            logger.critical(f"Error in worked: {traceback.format_exc()}", extra={"client_host": host})
            return {"Error": "Unknown error"}
    else:
        result_audio = dict[f"{str(host)}"]
        time_recognize_start = time.time()
        result = pipe(result_audio, generate_kwargs={"language": "russian"})
        p = th.Thread(target=saveAudioToFile, args=(host, time_query, result, result_audio))
        p.start()
        time_end = time.time()
        print("time for work: ", time_end - time_start)
        dict[f"{str(host)}"] = []
        return {'result': result["text"]}


app.mount('', StaticFiles(directory='./', html=True), name='static')

