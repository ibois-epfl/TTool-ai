# pull official base image
FROM python:3.11.2-slim-buster

# Install dependencies
COPY ./requirements.txt .
RUN pip install -r requirements.txt

# set work directory
WORKDIR /usr/src/app
