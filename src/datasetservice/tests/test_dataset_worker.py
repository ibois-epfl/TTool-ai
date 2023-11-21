import ctypes
import os
import pathlib
import random
import threading
import time
from unittest.mock import Mock, patch

import cv2
import dataset_worker
import numpy as np
import pika
import pytest
import sqlalchemy

random.seed(42)


@pytest.fixture
def video_file(tmp_path):
    video_path = tmp_path / "test_video.mp4"
    fps = 29.974512
    duration = 3
    size = (720, 1280)
    out = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size[::-1], False
    )
    for _ in range(int(fps * duration)):
        data = np.random.randint(0, 256, size, dtype="uint8")
        out.write(data)
    out.release()
    return video_path


def _check_processed_video(video_file):
    directory = video_file.parents[0]
    train_dir = directory / "train"
    val_dir = directory / "val"
    assert train_dir.is_dir()
    assert val_dir.is_dir()
    n_train = len(list(train_dir.glob("*.png")))
    n_val = len(list(val_dir.glob("*.png")))
    assert np.isclose(n_train / (n_val + n_train), 0.8, atol=0.05)


def test_process_video(video_file):
    dataset_worker.process_video(video_file)
    _check_processed_video(video_file)


@pytest.fixture
def mock_blocking_connection():
    with patch(
        "pika.BlockingConnection", spec=pika.BlockingConnection
    ) as mock_connection:
        yield mock_connection


@pytest.fixture
def mock_channel(mock_blocking_connection):
    mock_channel = mock_blocking_connection.return_value.channel.return_value
    return mock_channel


def test_worker_connects_and_receives_messages(mock_blocking_connection, mock_channel):
    mock_callback = Mock(spec=dataset_worker.Callback.callback)

    worker = dataset_worker.DatasetWorker(queue="test_queue")
    worker.connect(
        host="test_host", user="test_user", password="test_password", port=5672
    )

    worker.start_consuming(callback=mock_callback)

    # is_closed attribute has to be set for the mock connection,
    # because the close function is not called otherwise.
    worker.connection.is_closed = False
    worker.close_connection()

    credentials = pika.PlainCredentials(username="test_user", password="test_password")
    connection_params = pika.ConnectionParameters(
        host="test_host", port=5672, credentials=credentials
    )
    mock_blocking_connection.assert_called_once_with(connection_params)

    mock_channel.queue_declare.assert_called_once_with(queue="test_queue", durable=True)

    mock_channel.basic_qos.assert_called_once_with(prefetch_count=1)
    mock_channel.basic_consume.assert_called_once_with(
        queue="test_queue", on_message_callback=mock_callback
    )

    mock_channel.start_consuming.assert_called_once()
    mock_blocking_connection.return_value.close.assert_called_once()


@patch("sqlalchemy.create_engine", spec=sqlalchemy.create_engine)
@patch("sqlalchemy.orm.Session", spec=sqlalchemy.orm.Session)
@patch("dataset_worker.process_video", spec=dataset_worker.process_video)
def test_callback(mock_process_video, mock_session, mock_create_engine):
    path = "/tmp/test.mp4"
    body = str(path).encode("utf-8")
    callback = dataset_worker.Callback()
    callback.connect("URL")
    callback.callback(None, None, None, body)
    mock_process_video.assert_called_once_with(path=pathlib.Path(path))
    mock_create_engine.assert_called_once_with("URL")
    mock_session.assert_called_with(mock_create_engine.return_value)


def connect_to_rabbitmq(user, password, host, port, queue):
    credentials = pika.PlainCredentials(
        username=user,
        password=password,
    )
    connection_params = pika.ConnectionParameters(
        host=host,
        port=port,
        credentials=credentials,
    )

    for attempt in range(5):
        try:
            connection = pika.BlockingConnection(connection_params)
            channel = connection.channel()
            channel.queue_declare(queue=queue, durable=True)
            return channel
        except pika.exceptions.AMQPConnectionError as e:
            if attempt == 4:
                raise e
            else:
                time.sleep(1)
                continue


def get_postgres_url():
    pg_user = os.environ.get("POSTGRES_USER")
    pg_password = os.environ.get("POSTGRES_PASSWORD")
    pg_host = os.environ.get("POSTGRES_HOST")
    pg_port = os.environ.get("POSTGRES_PORT")
    pg_db = os.environ.get("POSTGRES_DB")
    postgres_url = f"postgresql://{pg_user}:{pg_password}@{pg_host}:{pg_port}/{pg_db}"
    return postgres_url


@pytest.fixture
def video_file_with_db_entry(video_file):
    database_url = get_postgres_url()
    engine = sqlalchemy.create_engine(database_url)
    dataset_worker.Base.metadata.create_all(bind=engine)
    with sqlalchemy.orm.Session(engine) as session:
        new_video = dataset_worker.VideoDB(
            label="bla",
            video_path=str(video_file),
            video_hash="123456",
            upload_status=dataset_worker.Status.PENDING,
        )
        session.add(new_video)
        session.commit()
    engine.dispose()
    return video_file


def test_integration_dataset_worker(video_file_with_db_entry):
    video_file = video_file_with_db_entry

    rbbt_user = os.environ.get("RABBITMQ_DEFAULT_USER")
    rbbt_password = os.environ.get("RABBITMQ_DEFAULT_PASS")
    rbbt_host = os.environ.get("RABBITMQ_HOST")
    rbbt_port = os.environ.get("RABBITMQ_PORT")
    rbbt_queue = os.environ.get("RABBITMQ_TEST_QUEUE")
    channel = connect_to_rabbitmq(
        rbbt_user, rbbt_password, rbbt_host, rbbt_port, rbbt_queue
    )

    # Remove any old messages
    channel.queue_purge(rbbt_queue)

    postgres_url = get_postgres_url()

    def run_worker():
        worker = dataset_worker.DatasetWorker(rbbt_queue)
        worker.connect(
            user=rbbt_user, password=rbbt_password, host=rbbt_host, port=rbbt_port
        )
        callback = dataset_worker.Callback()
        callback.connect(postgres_url)
        try:
            worker.start_consuming(callback=callback.callback)
        finally:
            callback.close_connection()

    worker_thread = threading.Thread(target=run_worker)
    worker_thread.start()

    properties = pika.BasicProperties(delivery_mode=pika.spec.PERSISTENT_DELIVERY_MODE)

    channel.basic_publish(
        exchange="",
        routing_key=rbbt_queue,
        body=str(video_file),
        properties=properties,
    )

    engine = sqlalchemy.create_engine(postgres_url)

    def get_status():
        with sqlalchemy.orm.Session(engine) as session:
            stmt = sqlalchemy.select(dataset_worker.VideoDB).where(
                dataset_worker.VideoDB.video_path == str(video_file)
            )
            status = session.scalars(stmt).one().upload_status
            return status

    status = get_status()
    while status in [dataset_worker.Status.PENDING, dataset_worker.Status.PROCESSING]:
        time.sleep(1)
        status = get_status()
    assert status == dataset_worker.Status.COMPLETED
    engine.dispose()

    _check_processed_video(video_file)

    # Kill the worker thread
    tid = worker_thread.ident
    exctype = ValueError
    ctypes.pythonapi.PyThreadState_SetAsyncExc(
        ctypes.c_long(tid), ctypes.py_object(exctype)
    )
