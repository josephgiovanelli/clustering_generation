FROM mcr.microsoft.com/vscode/devcontainers/python:0-3.9

RUN apt-get update && \
    apt-get install -y git --no-install-recommends

COPY requirements.txt requirements.txt

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

RUN rm requirements.txt

WORKDIR /home