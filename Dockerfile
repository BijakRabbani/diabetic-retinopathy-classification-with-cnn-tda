FROM mcr.microsoft.com/devcontainers/python:1-3.8-bookworm

WORKDIR /app

RUN sudo apt-get update

RUN sudo apt-get install -y libboost-all-dev

RUN sudo apt-get install -y cmake

RUN pip install --upgrade pip

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt