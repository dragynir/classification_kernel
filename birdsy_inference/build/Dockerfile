FROM ubuntu:20.04 AS darknet_build

RUN apt-get update -y && apt-get install -y python3-pip python3-dev


RUN apt-get install -y \
        wget \
        unzip \
        git

RUN pip3 install numpy

WORKDIR /build
COPY .. .

RUN sed -i 's/NUMPY=.*/NUMPY=1/' Makefile && \
        make

FROM ubuntu:20.04

RUN ln -snf /usr/share/zoneinfo/$CONTAINER_TIMEZONE /etc/localtime && echo $CONTAINER_TIMEZONE > /etc/timezone
RUN apt-get update -y && apt-get install -y python3-pip python3-dev wget tzdata




RUN pip3 install numpy pillow pika requests boto3 validators opencv-python albumentations>=1.1.0 pytorch_lightning>=1.4.9 efficientnet_pytorch>=0.7.1 fire>=0.4.0
RUN pip3 install addict>=2.4.0 wandb>=0.12.4 pandas>=1.3.3 torchvision>=0.10.1 seaborn>=0.11.2 shap>=0.39.0
RUN apt-get update
RUN apt-get install -y libgl1-mesa-glx libgtk2.0-dev

WORKDIR /app

COPY ../classify.py .
COPY ../inference_docker.py .
COPY python ./python
COPY yolo ./yolo
COPY ../source ./source
