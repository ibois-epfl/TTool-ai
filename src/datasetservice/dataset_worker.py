import os
import pathlib
import random
import time

import cv2
import pika


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


def callback(ch, method, properties, body):
    path_str = body.decode("utf-8")
    path = pathlib.Path(path_str)
    process_video(path)


class DatasetWorker:
    def __init__(self, queue):
        self.queue = queue
        self.connection = None
        self.channel = None

    def connect(self, user, password, host, port):
        credentials = pika.PlainCredentials(
            username=user,
            password=password,
        )
        connection_params = pika.ConnectionParameters(
            host=host,
            port=port,
            credentials=credentials,
        )
        self.connection = pika.BlockingConnection(connection_params)
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue=self.queue, durable=True)

    def start_consuming(self, callback):
        self.channel.basic_qos(prefetch_count=1)
        self.channel.basic_consume(queue=self.queue, on_message_callback=callback)
        self.channel.start_consuming()

    def close_connection(self):
        if self.connection and not self.connection.is_closed:
            self.connection.close()


if __name__ == "__main__":
    user = os.environ.get("RABBITMQ_DEFAULT_USER")
    password = os.environ.get("RABBITMQ_DEFAULT_PASS")
    host = os.environ.get("RABBITMQ_HOST")
    port = os.environ.get("RABBITMQ_PORT")
    queue = os.environ.get("RABBITMQ_TEST_QUEUE")

    worker = DatasetWorker(queue)

    # The rabbitmq container for testing sometimes needs
    # more time to start so we try to connect multiple times.
    while True:
        try:
            worker.connect(user, password, host, port)
            print("Connection to RabbitMQ succeeded.")
            break
        except pika.exceptions.AMQPConnectionError:
            print("Connection to RabbitMQ failed.")
            time.sleep(1)
            continue

    try:
        worker.start_consuming(callback=process_video)
    finally:
        worker.close_connection()
