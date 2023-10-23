import celery
import fastapi
from pydantic import BaseModel

import worker


class TrainingParams(BaseModel):
    max_epochs: int = 45
    batch_size: int = 90
    training_data: tuple = ()
    validation_data: tuple = ()


app = fastapi.FastAPI()


@app.post("/trainings", status_code=201)
def train(training_params: TrainingParams):
    task = worker.train.delay(**dict(training_params))
    return fastapi.responses.JSONResponse({"training_id": task.id})


@app.get("/trainings/{task_id}")
def get_status(task_id):
    task_result = celery.result.AsyncResult(task_id)
    result = {
        "training_id": task_id,
        "training_status": task_result.status,
        "training_result": task_result.result,
    }
    return fastapi.responses.JSONResponse(result)
