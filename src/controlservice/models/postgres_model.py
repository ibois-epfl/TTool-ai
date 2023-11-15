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
    upload_status = Column(String, default=DataLoaderStatus.PENDING)


def serialize_video_db(video):
    return {
        "id": video.id,
        "label": video.label,
        "video_path": video.video_path,
        "video_hash": video.video_hash,
        "upload_status": video.upload_status
    }
