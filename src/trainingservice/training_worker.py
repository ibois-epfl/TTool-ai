import json
import os
import pathlib
import time

import pika
import sqlalchemy
import sqlalchemy.orm

import training


class Base(sqlalchemy.orm.DeclarativeBase):
    pass


class Status(str, sqlalchemy.Enum):
    PENDING = "pending"
    TRAINING = "training"
    DONE = "done"
    FAILED = "failed"


class TrainingDB(Base):
    __tablename__ = "trainings"
    id = sqlalchemy.Column(sqlalchemy.Integer, primary_key=True, index=True)

    data_dirs = sqlalchemy.Column(sqlalchemy.String)
    max_epochs = sqlalchemy.Column(sqlalchemy.Integer)
    batch_size = sqlalchemy.Column(sqlalchemy.Integer)

    training_hash = sqlalchemy.Column(sqlalchemy.String, unique=True)

    status = sqlalchemy.Column(sqlalchemy.String, default=Status.PENDING)
    logdir = sqlalchemy.Column(sqlalchemy.String)
    weights = sqlalchemy.Column(sqlalchemy.String)
    trace_file = sqlalchemy.Column(sqlalchemy.String)


class TrainingParams:
    def __init__(self, bstr):
        param_dict = json.loads(bstr.decode("utf-8"))
        self.max_epochs = param_dict["max_epochs"]
        self.batch_size = param_dict["batch_size"]
        self.data_dirs = tuple(param_dict["data_dirs"])

    def __hash__(self):
        return hash((self.max_epochs, self.batch_size, *self.data_dirs))


class Callback:
    def __init__(self):
        self.engine = None

    def connect(self, database_url):
        self.engine = sqlalchemy.create_engine(database_url)
        Base.metadata.create_all(self.engine)

    def _add_info(
        self, training_hash, log_dir=None, trace_file=None, weights=None, status=None
    ):
        stmt = sqlalchemy.select(TrainingDB).where(
            TrainingDB.training_hash == training_hash
        )
        with sqlalchemy.orm.Session(self.engine) as session:
            training_entry = session.scalars(stmt).one()
            if log_dir is not None:
                training_entry.log_dir = log_dir
            if trace_file is not None:
                training_entry.trace_file = trace_file
            if weights is not None:
                training_entry.weights = weights
            if status is not None:
                training_entry.status = status
            session.commit()

    def callback(self, ch, method, properties, body):
        training_params = TrainingParams(body)
        training_hash = hash(training_params)
        new_training = TrainingDB(
            max_epochs=training_params.max_epochs,
            data_dirs=training_params.data_dirs,
            training_hash=training_hash,
        )
        with sqlalchemy.orm.Session(self.engine) as session:
            session.add(new_training)
            session.commit()

        log_dir = pathlib.Path(f"/data/{training_hash}")
        log_dir.mkdir()
        self._add_info(training_hash, log_dir=log_dir)
        # try:
        self._add_info(training_hash, status=Status.TRAINING)
        weights_file, trace_file = training.train(
            max_epochs=training_params.max_epochs,
            batch_size=training_params.batch_size,
            data_dirs=training_params.data_dirs,
            log_dir=log_dir,
        )
        self._add_info(
            training_hash,
            status=Status.DONE,
            weights=weights_file,
            trace_file=trace_file,
        )
        # except Exception as e:
        #     self._add_info(training_hash, status=Status.FAILED)

    def close_connection(self):
        self.engine.dispose()


class TrainingWorker:
    def __init__(self, queue):
        pass

    def connect(self, user, password, host, port):
        pass

    def start_consuming(self, callback):
        pass

    def close_connection(self):
        pass


if __name__ == "__main__":
    USER = os.environ.get("RABBITMQ_DEFAULT_USER")
    PASSWORD = os.environ.get("RABBITMQ_DEFAULT_PASS")
    HOST = os.environ.get("RABBITMQ_HOST")
    PORT = os.environ.get("RABBITMQ_PORT")
    QUEUE = os.environ.get("RABBITMQ_DATA_QUEUE")

    worker = TrainingWorker(QUEUE)

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
    time.sleep(3600)

    try:
        worker.start_consuming(callback=callback.callback)
    finally:
        worker.close_connection()
        callback.close_connection()
