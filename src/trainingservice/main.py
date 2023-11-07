import os

import celery
import fastapi
import psycopg2
from pydantic import BaseModel

import worker

postgres_url = os.environ.get("POSTGRES_URL")
conn = psycopg2.connect(postgres_url)
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


@app.get("/trainings/{task_id}")
def get_status(task_id):
    task_result = celery.result.AsyncResult(task_id)
    result = {
        "training_id": task_id,
        "training_status": task_result.status,
        "training_result": task_result.result,
    }
    return fastapi.responses.JSONResponse(result)
