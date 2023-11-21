class Callback:
    def __init__(self):
        pass

    def connect(self, url):
        pass

    def callback(self, ch, method, properties, body):
        pass

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

    try:
        worker.start_consuming(callback=callback.callback)
    finally:
        worker.close_connection()
        callback.close_connection()