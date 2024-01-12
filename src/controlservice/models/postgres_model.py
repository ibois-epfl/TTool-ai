import sqlalchemy
from sqlalchemy import Column, Integer, String, Enum
from config.postgres_config import Base


class Status(Enum):
    PENDING = 'pending'
    PROCESSING = 'processing'
    TRAINING = 'training'
    COMPLETED = 'completed'
    FAILED = 'failed'


class VideoDB(Base):
    __tablename__ = 'videos'
    id = sqlalchemy.Column(Integer, primary_key=True, index=True)
    label = sqlalchemy.Column(String)
    video_path = sqlalchemy.Column(String)
    video_hash = sqlalchemy.Column(String, unique=True)
    data_dir = sqlalchemy.Column(String)
    upload_status = sqlalchemy.Column(String, default=Status.PENDING)

class TrainDB(Base):
    __tablename__ = "train"
    id = sqlalchemy.Column(sqlalchemy.Integer, primary_key=True, index=True)

    data_dirs = sqlalchemy.Column(sqlalchemy.String)
    max_epochs = sqlalchemy.Column(sqlalchemy.Integer)
    batch_size = sqlalchemy.Column(sqlalchemy.Integer)

    user_id = sqlalchemy.Column(sqlalchemy.String)
    classes = sqlalchemy.Column(sqlalchemy.JSON)

    training_hash = sqlalchemy.Column(sqlalchemy.BigInteger, unique=True)

    status = sqlalchemy.Column(sqlalchemy.String, default=Status.PENDING)
    log_dir = sqlalchemy.Column(sqlalchemy.String)
    weights = sqlalchemy.Column(sqlalchemy.String)
    trace_file = sqlalchemy.Column(sqlalchemy.String)
    label_map_file = sqlalchemy.Column(sqlalchemy.String)