# Запуск


### В Docker
Для запуска нужно ввести команду:
```bash
docker build -t vice-reconizer . &&  docker run -p 8081:8081 -v save-audio:/code/save-audio vice-reconizer
```
### Через консоль
Для запуска нужно ввести команду:
```bash
uvicorn main:app --port 8081 --reload 
```