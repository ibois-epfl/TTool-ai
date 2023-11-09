import os

from celery import Celery

import training

rabbitmq_user = os.environ.get("RABBITMQ_DEFAULT_USER")
rabbitmq_password = os.environ.get("RABBITMQ_DEFAULT_PASS")
rabbitmq_host = os.environ.get("RABBITMQ_HOST")
rabbitmq_port = os.environ.get("RABBITMQ_PORT")
train_queue = os.environ.get("RABBITMQ_TRAIN_QUEUE")

celery = Celery(__name__)
celery.conf.broker_url = (
    f"amqp://{rabbitmq_user}:{rabbitmq_password}@{rabbitmq_host}:{rabbitmq_port}"
)
celery.conf.task_routes = {"train": {"queue": train_queue}}


@celery.task(name="train")
def train(**kwargs):
    training.train(**kwargs)
    return True
