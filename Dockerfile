FROM mcr.microsoft.com/vscode/devcontainers/python:0-3.9

RUN cd home && mkdir dump
WORKDIR /home/dump
COPY resources resources
COPY scripts scripts
COPY src src
COPY requirements.txt requirements.txt

RUN apt-get update && \
    apt-get install -y git --no-install-recommends

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

RUN mkdir results
RUN chmod 777 scripts/*
ENTRYPOINT ["./scripts/wrapper_experiments.sh"]