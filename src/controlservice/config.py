from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import declarative_base
import pika
import os

# Docker Video Directory
DOCKER_VIDEO_DIR = "/media/train_videos/"


# Postgres Config
POSTGRES_DB = os.getenv('POSTGRES_DB')
POSTGRES_USER = os.getenv('POSTGRES_USER')
POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD')
DATABASE_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@postgres/{POSTGRES_DB}"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def init_db():
    Base.metadata.create_all(bind=engine)


# RabbitMQ Config
RABBITMQ_PORT = int(os.getenv('RABBITMQ_PORT'))
RABBITMQ_DEFAULT_USER = os.getenv('RABBITMQ_DEFAULT_USER')
RABBITMQ_DEFAULT_PASS = os.getenv('RABBITMQ_DEFAULT_PASS')
RABBITMQ_TEST_QUEUE = os.getenv('RABBITMQ_TEST_QUEUE')
RABBITMQ_DATA_QUEUE = os.getenv('RABBITMQ_DATA_QUEUE')
RABBITMQ_TRAIN_QUEUE = os.getenv('RABBITMQ_TRAIN_QUEUE')
producer_credentials = pika.PlainCredentials(RABBITMQ_DEFAULT_USER, RABBITMQ_DEFAULT_PASS)
producer_parameters = pika.ConnectionParameters(
                host='rabbitmq', port=RABBITMQ_PORT, credentials=producer_credentials,
                heartbeat=0, blocked_connection_timeout=300)
producer_rabbit_connection = pika.BlockingConnection(producer_parameters)
producer_rabbit_channel = producer_rabbit_connection.channel()
