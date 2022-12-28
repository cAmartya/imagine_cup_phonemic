FROM python:3.8-slim-buster

RUN apt-get update -y
RUN apt install libgl1-mesa-glx -y
RUN apt-get install 'ffmpeg'\
    'libsm6'\
    'libxext6'  -y

COPY requirements.txt requirements.txt
RUN pip3 install --upgrade pip
RUN pip3 install opencv-python==4.3.0.38
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5000
ENTRYPOINT ["python"]

CMD ["app.py"]