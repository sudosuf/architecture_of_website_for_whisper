# Запуск


### В Docker
Для запуска нужно ввести команду:
```bash
docker build -t vice-reconizer path/to/work/direrctory && docker run vice-reconizer
```
### Через консоль
Для запуска нужно ввести команду:
```bash
uvicorn main:app --port 8081 --reload 
```