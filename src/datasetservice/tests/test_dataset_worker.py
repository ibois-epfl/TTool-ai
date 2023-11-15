import pathlib
import random
from unittest.mock import Mock, patch

import cv2
import dataset_worker
import numpy as np
import pika
import pytest

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
    mock_callback = Mock(spec=dataset_worker.callback)

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


@patch("dataset_worker.process_video", spec=dataset_worker.process_video)
def test_callback(process_video):
    path = "/tmp/test.mp4"
    body = str(path).encode("utf-8")
    dataset_worker.callback(None, None, None, body)
    process_video.assert_called_once_with(path=pathlib.Path(path))


# def test_connect_worker_to_queue(video_file):
#     user = os.environ.get("RABBITMQ_DEFAULT_USER")
#     password = os.environ.get("RABBITMQ_DEFAULT_PASS")
#     host = os.environ.get("RABBITMQ_HOST")
#     port = os.environ.get("RABBITMQ_PORT")
#     queue = os.environ.get("RABBITMQ_TEST_QUEUE")
#
#     credentials = pika.PlainCredentials(
#         username=user,
#         password=password,
#     )
#     connection_params = pika.ConnectionParameters(
#         host=host,
#         port=port,
#         credentials=credentials,
#     )
#
#     for attempt in range(5):
#         try:
#             connection = pika.BlockingConnection(connection_params)
#             break
#         except pika.exceptions.AMQPConnectionError:
#             time.sleep(1)
#             continue
#
#     channel = connection.channel()
#     channel.queue_declare(queue=queue, durable=True)
#
#     properties = pika.BasicProperties(
#             delivery_mode=pika.spec.PERSISTENT_DELIVERY_MODE
#       )
#
#     channel.basic_publish(
#         exchange="",
#         routing_key=queue,
#         body=str(video_file),
#         properties=properties,
#     )
#
#     worker_channel = worker.connect_worker_to_queue(
#         user,
#         password,
#         host,
#         port,
#         queue,
#     )
#
#     method_frame, properties, body = worker_channel.basic_get(queue=queue)
#     print(method_frame, properties, body)
#     # for (
#     #     method_frame,
#     #     properties,
#     #     body,
#     # ) in worker_channel.basic_get(queue=queue):
#     #     print(method_frame, properties, body)
#     # worker.callback(worker_channel, method_frame, properties, body)
#     # _check_processed_video(video_file)
