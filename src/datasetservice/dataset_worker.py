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
    data_dir = sqlalchemy.Column(sqlalchemy.String)
    upload_status = sqlalchemy.Column(sqlalchemy.String, default=Status.PENDING)


def process_video(path: pathlib.Path):
    directory = path.parents[0]
    train_dir = directory / "train"
    val_dir = directory / "val"
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
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

    def _update_status(self, path, status):
        session = sqlalchemy.orm.Session(self.engine)
        stmt = sqlalchemy.select(VideoDB).where(VideoDB.video_path == path)
        video_entry = session.scalars(stmt).one()
        video_entry.upload_status = status
        session.commit()
        session.close()

    def callback(self, ch, method, properties, body):
        path_str = body.decode("utf-8")
        try:
            self._update_status(path_str, Status.PROCESSING)
            path = pathlib.Path(path_str)
            print(path)
            process_video(path)
            self._update_status(path_str, Status.COMPLETED)
        except Exception as e:
            self._update_status(path_str, Status.FAILED)
            print(e)

    def close_connection(self):
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
            heartbeat=0,
            blocked_connection_timeout=300
        )
        self.connection = pika.BlockingConnection(connection_params)
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue=self.queue, durable=True)

    def start_consuming(self, callback):
        self.channel.basic_qos(prefetch_count=1)
        self.channel.basic_consume(
            queue=self.queue, on_message_callback=callback, auto_ack=True
        )
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
