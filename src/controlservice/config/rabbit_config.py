import pika
import os

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
