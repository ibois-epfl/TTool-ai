import os
import pathlib
import random

import cv2
from celery import Celery

rabbitmq_user = os.environ.get("RABBITMQ_DEFAULT_USER")
rabbitmq_password = os.environ.get("RABBITMQ_DEFAULT_PASS")
rabbitmq_host = os.environ.get("RABBITMQ_HOST")
rabbitmq_port = os.environ.get("RABBITMQ_PORT")
data_queue = os.environ.get("RABBITMQ_DATA_QUEUE")

celery = Celery(__name__)
celery.conf.broker_url = (
    f"amqp://{rabbitmq_user}:{rabbitmq_password}@{rabbitmq_host}:{rabbitmq_port}"
)
celery.conf.task_routes = {"process_video": {"queue": data_queue}}


@celery.task(name="process_video")
def process_video(path: pathlib.Path):
    directory = path.parents[0]
    train_dir = directory / "train"
    val_dir = directory / "val"
    train_dir.mkdir()
    val_dir.mkdir()
    cap = cv2.VideoCapture(str(path))

    if not cap.isOpened():
        raise RuntimeError("Could not open capture.")

    frame_idx = 0
    while cap.isOpened():
        if random.random() < 0.8:
            out_dir = train_dir
        else:
            out_dir = val_dir
        ret, frame = cap.read()
        if ret:
            out_file = out_dir / f"{frame_idx}.png"
            cv2.imwrite(str(out_file), frame)
            frame_idx += 1
        else:
            break
    cap.release()
    return True
