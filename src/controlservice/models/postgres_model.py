from sqlalchemy import Column, Integer, String, Enum
from config.postgres_config import Base


class DataLoaderStatus(Enum):
    PENDING = 'pending'
    PROCESSING = 'processing'
    COMPLETED = 'completed'
    FAILED = 'failed'


class VideoDB(Base):
    __tablename__ = 'videos'
    id = Column(Integer, primary_key=True, index=True)
    label = Column(String)
    video_path = Column(String)
    video_hash = Column(String, unique=True)
    data_dir = Column(String)
    upload_status = Column(String, default=DataLoaderStatus.PENDING)

class TrainDB(Base):
    __tablename__ = 'train'
    id = Column(Integer, primary_key=True, index=True)
    labels = Column(String)
    model_path = Column(String)
    train_status = Column(String, default=DataLoaderStatus.PENDING)
    log_dir = Column(String)