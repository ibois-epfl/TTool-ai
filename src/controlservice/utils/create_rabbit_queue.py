import pika

credentials = pika.PlainCredentials('ttoolai', '123456')

parameters = pika.ConnectionParameters(
        '128.178.91.167', 5672, 'ttoolai', credentials, heartbeat=600, blocked_connection_timeout=300)

connection = pika.BlockingConnection(parameters)
channel = connection.channel()

# Declare a queue
channel.queue_declare(queue='test', durable=True)

connection.close()