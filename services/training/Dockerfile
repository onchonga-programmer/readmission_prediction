FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --timeout=300 -r requirements.txt

COPY train.py .

CMD ["python", "train.py"]