FROM python:3.12
WORKDIR /code
RUN apt-get update && apt-get install -y libsndfile1
COPY requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
COPY . /code
EXPOSE 8081
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8081",  "--log-level", "trace"]