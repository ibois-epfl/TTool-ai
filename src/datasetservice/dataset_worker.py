import os
import pathlib
import random
import time

import cv2
import pika
import sqlalchemy
import sqlalchemy.orm


class Base(sqlalchemy.orm.DeclarativeBase):
    pass


class Status(str, sqlalchemy.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class VideoDB(Base):
    __tablename__ = "videos"
    id = sqlalchemy.Column(sqlalchemy.Integer, primary_key=True, index=True)
    label = sqlalchemy.Column(sqlalchemy.String)
    video_path = sqlalchemy.Column(sqlalchemy.String)
    video_hash = sqlalchemy.Column(sqlalchemy.String, unique=True)
    upload_status = sqlalchemy.Column(sqlalchemy.String, default=Status.PENDING)


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


class Callback:
    def __init__(self):
        self.engine = None
        self.session = None

    def connect(self, database_url):
        self.engine = sqlalchemy.create_engine(database_url)
        Base.metadata.create_all(self.engine)
        self.session = sqlalchemy.orm.Session(self.engine)

    def callback(self, ch, method, properties, body):
        path_str = body.decode("utf-8")
        stmt = sqlalchemy.select(VideoDB).where(VideoDB.video_path == path_str)
        video_entry = self.session.scalars(stmt).one()
        try:
            video_entry.upload_status = Status.PROCESSING
            self.session.commit()
            path = pathlib.Path(path_str)
            process_video(path)
            video_entry.upload_status = Status.COMPLETED
            self.session.commit()
        except Exception:
            video_entry.upload_status = Status.FAILED
            self.session.commit()

    def close_connection(self):
        self.session.close()
        self.engine.dispose()


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
    USER = os.environ.get("RABBITMQ_DEFAULT_USER")
    PASSWORD = os.environ.get("RABBITMQ_DEFAULT_PASS")
    HOST = os.environ.get("RABBITMQ_HOST")
    PORT = os.environ.get("RABBITMQ_PORT")
    QUEUE = os.environ.get("RABBITMQ_DATA_QUEUE")

    worker = DatasetWorker(QUEUE)

    # The rabbitmq container for testing sometimes needs
    # more time to start so we try to connect multiple times.
    while True:
        try:
            worker.connect(user=USER, password=PASSWORD, host=HOST, port=PORT)
            print("Connection to RabbitMQ succeeded.")
            break
        except pika.exceptions.AMQPConnectionError:
            print("Connection to RabbitMQ failed.")
            time.sleep(1)
            continue

    # Connect to postgres db
    USER = os.environ.get("POSTGRES_USER")
    PASSWORD = os.environ.get("POSTGRES_PASSWORD")
    HOST = os.environ.get("POSTGRES_HOST")
    PORT = os.environ.get("POSTGRES_PORT")
    DB = os.environ.get("POSTGRES_DB")
    DB_URL = f"postgresql://{USER}:{PASSWORD}@{HOST}:{PORT}/{DB}"

    callback = Callback()
    callback.connect(DB_URL)

    try:
        worker.start_consuming(callback=callback.callback)
    finally:
        worker.close_connection()
        callback.close_connection()
