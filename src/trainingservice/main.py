import os

import fastapi
import psycopg2
from pydantic import BaseModel

import worker

postgres_db = os.environ.get("POSTGRES_DB")
postgres_user = os.environ.get("POSTGRES_USER")
postgres_password = os.environ.get("POSTGRES_PASSWORD")
postgres_host = os.environ.get("POSTGRES_HOST")
postgres_port = os.environ.get("POSTGRES_PORT")

conn = psycopg2.connect(
    dbname=postgres_db,
    user=postgres_user,
    password=postgres_password,
    host=postgres_host,
    port=postgres_port,
)

curs = conn.cursor()
tables = []
curs.execute(
    "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
)
for table in curs.fetchall():
    tables.append(table[0])
if "trainings" not in tables:
    curs.execute(
        """CREATE TABLE trainings (training_id varchar(255), max_epochs int,
        batch_size int, datasets varchar(255), log_dir varchar(255),
        weights_path varchar(255), celery_task_id varchar(255))"""
    )
curs.close()
conn.commit()


class TrainingParams(BaseModel):
    max_epochs: int = 45
    batch_size: int = 90
    data: tuple = ()

    def __hash__(self):
        return hash((self.max_epochs, self.batch_size, *self.data))


app = fastapi.FastAPI()


@app.get("/trainings")
def get_list_of_traingins():
    curs = conn.cursor()
    curs.execute(
        """SELECT training_id, datasets, batch_size,
            max_epochs FROM trainings"""
    )
    results = curs.fetchall()
    curs.close()
    print(results)
    return fastapi.responses.JSONResponse(results)


@app.post("/trainings", status_code=201)
def train(training_params: TrainingParams):
    task = worker.train.delay(**dict(training_params))
    training_id = hash(training_params)
    max_epochs = training_params.max_epochs
    batch_size = training_params.batch_size
    # data = str(training_params.data)
    curs = conn.cursor()
    curs.execute(
        f"""INSERT INTO trainings (training_id, max_epochs, batch_size,
        celery_task_id) VALUES ({training_id}, {max_epochs}, {batch_size},
        '{task.id}')"""
    )
    curs.close()
    conn.commit()
    return fastapi.responses.JSONResponse({"training_id": training_id})


@app.get("/trainings/{training_id}")
def get_status(training_id):
    curs = conn.cursor()
    curs.execute(
        f"""SELECT training_id, datasets, batch_size,
            max_epochs FROM trainings WHERE training_id='{training_id}'"""
    )
    result = curs.fetchall()
    curs.close()
    return fastapi.responses.JSONResponse(result)
