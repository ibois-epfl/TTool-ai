# pull official base image
FROM nvidia/cuda:11.4.3-runtime-ubuntu20.04

# install python
RUN apt-get update
RUN apt-get install -y --no-install-recommends python3.9 python3-pip

# set work directory
WORKDIR /usr/src/app

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# install dependencies
RUN pip install --upgrade pip
COPY ./requirements.txt .
RUN pip install -r requirements.txt

# copy project
COPY . .
