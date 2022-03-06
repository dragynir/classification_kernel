FROM continuumio/miniconda3

RUN apt-get update -y

RUN conda install -y nomkl bokeh numpy pandas

COPY . /app
WORKDIR /app 

VOLUME '/app/data'

EXPOSE 5006
CMD ["bokeh", "serve", "--show", "app.py"]

# docker exec -it [container-id] bash
# docker build -t app_ui .
# type=bind,source="$(pwd)"/target,target=/app \
# docker run -p 5006:5006 -it --mount type=bind,source=C:\Users\dkoro\PythonProjects\Birdsy\birdsy_cl\inference\monitoring\ui\data,target=/app/data app_ui