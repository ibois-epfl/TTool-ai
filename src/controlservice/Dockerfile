FROM ubuntu:20.04

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update
RUN apt install -y software-properties-common
RUN apt install -y python3.9
RUN apt install -y python3-pip

RUN mkdir /controlservice
COPY . /controlservice
RUN pip install -r /controlservice/requirements.txt
WORKDIR /controlservice/
