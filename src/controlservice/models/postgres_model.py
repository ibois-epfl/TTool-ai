from sqlalchemy import Column, Integer, String, Enum
from config.postgres_config import Base


class VideoStatus(str, Enum):
    PENDING = 'pending'
    PROCESSING = 'processing'
    COMPLETED = 'completed'
    FAILED = 'failed'


class VideoDB(Base):
    __tablename__ = 'videos'
    id = Column(Integer, primary_key=True, index=True)
    label = Column(String)
    video_path = Column(String)
    video_hash = Column(String)
    upload_status = VideoStatus.PENDING



