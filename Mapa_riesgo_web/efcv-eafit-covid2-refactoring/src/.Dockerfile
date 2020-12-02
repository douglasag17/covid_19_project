FROM tensorflow/tensorflow:latest

COPY . /app

WORKDIR /usr/src/app

CMD python /app/execute_pipe_line.py