import ctypes
import json
import os
import pathlib
import threading
import time
from unittest.mock import Mock, patch

import pika
import pytest
import sqlalchemy
import training
from training_worker import (
    Base,
    Callback,
    Status,
    TrainingDB,
    TrainingParams,
    TrainingWorker,
)


@pytest.fixture(params=("list", "list_of_lists"))
def data_dirs_list(request):
    if request.param == "list":
        data_dirs = ["/data/dir1", "/data/dir2", "/data/dir3"]
        expected = tuple(data_dirs)
    elif request.param == "list_of_lists":
        data_dirs = [
            ["/data/dir1", "/data/dir2"],
            ["/data/dir3", "/data/dir4", "/data/dir5"],
        ]
        expected = (
            "/data/dir1",
            "/data/dir2",
            "/data/dir3",
            "/data/dir4",
            "/data/dir5",
        )
    return data_dirs, expected


@pytest.fixture
def message_body(data_dirs_list):
    list_data_dirs, tuple_data_dirs = data_dirs_list
    param_dict = {
        "user_id": 1,
        "max_epochs": 10,
        "batch_size": 5,
        "data_dirs": list_data_dirs,
    }
    body = json.dumps(param_dict).encode("utf-8")
    expected = param_dict
    expected["data_dirs"] = tuple_data_dirs
    expected["hash"] = hash(
        (expected["max_epochs"], expected["batch_size"], *expected["data_dirs"])
    )
    return body, expected


def test_TrainingParams(message_body):
    body, expected = message_body
    training_params = TrainingParams(body)

    assert training_params.user_id == expected["user_id"]
    assert training_params.max_epochs == expected["max_epochs"]
    assert training_params.batch_size == expected["batch_size"]
    assert training_params.data_dirs == expected["data_dirs"]
    training_hash = hash(training_params)
    assert training_hash == expected["hash"]


@patch("training_worker.Base.metadata.create_all", spec=Base.metadata.create_all)
@patch("training_worker.TrainingParams", spec=TrainingParams)
@patch("training_worker.TrainingDB", spec=TrainingDB)
@patch("sqlalchemy.select", spec=sqlalchemy.select)
@patch("sqlalchemy.create_engine", spec=sqlalchemy.create_engine)
@patch("sqlalchemy.orm.Session", spec=sqlalchemy.orm.Session)
@patch("training.train", spec=training.train)
def test_Callback(
    mock_train,
    mock_session,
    mock_create_engine,
    mock_select,
    mock_training_db,
    mock_training_params,
    mock_create_all,
    message_body,
):
    # Setup
    body, expected = message_body
    mock_training_params.return_value.max_epochs = expected["max_epochs"]
    mock_training_params.return_value.batch_size = expected["batch_size"]
    mock_training_params.return_value.data_dirs = expected["data_dirs"]
    mock_training_params.return_value.user_id = expected["user_id"]
    mock_training_params.return_value.__hash__.return_value = expected["hash"]
    mock_training_params.reset_mock()

    mock_train.return_value = ("weights.pth", "trace.pt")

    # Run
    callback = Callback()
    callback.connect("URL")
    callback.callback(None, None, None, body)
    callback.close_connection()

    # Assertions
    mock_create_all.assert_called_once_with(mock_create_engine.return_value)
    mock_training_params.assert_called_once_with(body)

    mock_training_db.assert_called_once_with(
        max_epochs=expected["max_epochs"],
        batch_size=expected["batch_size"],
        data_dirs=expected["data_dirs"],
        user_id=expected["user_id"],
        training_hash=expected["hash"],
    )

    log_dir = pathlib.Path(f"/data/{expected['hash']}")
    assert log_dir.is_dir()

    mock_train.assert_called_once_with(
        max_epochs=expected["max_epochs"],
        batch_size=expected["batch_size"],
        data_dirs=expected["data_dirs"],
        log_dir=log_dir,
    )

    mock_create_engine.assert_called_once_with("URL")
    mock_session.assert_called_with(mock_create_engine.return_value)
    mock_select.assert_called_with(mock_training_db)


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


def test_training_worker(mock_blocking_connection, mock_channel):
    mock_callback = Mock(spec=Callback.callback)
    credentials = pika.PlainCredentials(username="test_user", password="test_password")
    connection_params = pika.ConnectionParameters(
        host="test_host", port=5672, credentials=credentials
    )

    # Run
    worker = TrainingWorker(queue="test_queue")
    worker.connect(
        user="test_user",
        password="test_password",
        host="test_host",
        port=5672,
    )
    worker.start_consuming(callback=mock_callback)
    # Emulate the behaviour of the real connection
    worker.connection.is_closed = False

    worker.close_connection()

    # Asserts
    mock_blocking_connection.assert_called_once_with(connection_params)
    mock_channel.queue_declare.assert_called_once_with(queue="test_queue", durable=True)
    mock_channel.basic_qos.assert_called_once_with(prefetch_count=1)
    mock_channel.basic_consume.assert_called_once_with(
        queue="test_queue", on_message_callback=mock_callback
    )
    mock_channel.start_consuming.assert_called_once()
    mock_blocking_connection.return_value.close.assert_called_once()


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


def test_integration_training_worker(data_dirs):
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
        worker = TrainingWorker(rbbt_queue)
        worker.connect(
            user=rbbt_user, password=rbbt_password, host=rbbt_host, port=rbbt_port
        )
        callback = Callback()
        callback.connect(postgres_url)
        try:
            worker.start_consuming(callback=callback.callback)
        finally:
            callback.close_connection()

    worker_thread = threading.Thread(target=run_worker)
    worker_thread.start()

    try:
        properties = pika.BasicProperties(
            delivery_mode=pika.spec.PERSISTENT_DELIVERY_MODE
        )

        training_params = {
            "user_id": 1,
            "max_epochs": 10,
            "batch_size": 5,
            "data_dirs": data_dirs,
        }
        body = json.dumps(training_params).encode("utf-8")
        training_hash = hash(TrainingParams(body))
        channel.basic_publish(
            exchange="",
            routing_key=rbbt_queue,
            body=body,
            properties=properties,
        )

        engine = sqlalchemy.create_engine(postgres_url)

        def get_status():
            with sqlalchemy.orm.Session(engine) as session:
                stmt = sqlalchemy.select(TrainingDB).where(
                    TrainingDB.training_hash == training_hash
                )
                status = session.scalars(stmt).one().status
                return status

        # Give the worker some time to create the Postgres entry
        time.sleep(5)

        status = get_status()
        while status in [Status.PENDING, Status.TRAINING]:
            time.sleep(1)
            status = get_status()
        assert status == Status.DONE
        engine.dispose()

        with sqlalchemy.orm.Session(engine) as session:
            stmt = sqlalchemy.select(TrainingDB).where(
                TrainingDB.training_hash == training_hash
            )
            entry = session.scalars(stmt).one()
            log_dir = entry.log_dir
            weights = entry.weights
            trace_file = entry.trace_file

        assert pathlib.Path(log_dir).is_dir()
        assert pathlib.Path(weights).exists()
        assert pathlib.Path(trace_file).exists()

    finally:
        # Kill the worker thread
        tid = worker_thread.ident
        exctype = ValueError
        ctypes.pythonapi.PyThreadState_SetAsyncExc(
            ctypes.c_long(tid), ctypes.py_object(exctype)
        )
