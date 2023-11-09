import hashlib
import os
import pathlib

import celery
import fastapi
import psycopg2

import worker

data_dir = pathlib.Path(os.environ.get("DATA_DIR"))

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
if "datasets" not in tables:
    curs.execute(
        """CREATE TABLE datasets (dataset_id varchar(255), label varchar(255),
        video_path varchar(255), video_hash varchar(255), n_images int,
        n_train_images int, n_val_images int, celery_task_id varchar(255))"""
    )
curs.close()
conn.commit()


app = fastapi.FastAPI()


@app.get("/datasets")
def get_list_of_datasets():
    curs = conn.cursor()
    curs.execute(
        """SELECT dataset_id, label, video_path,
            n_images, n_train_images, n_val_images FROM datasets"""
    )
    curs.close()
    conn.commit()
    results = curs.fetchall()
    return fastapi.responses.JSONResponse(results)


@app.post("/datasets")
async def upload_video(video: fastapi.UploadFile, label: str, status_code=201):
    # Temporarily save file until we have the hash
    tmp_dir = data_dir / "tmp_dir"
    tmp_dir.mkdir()
    target_path = tmp_dir / video.filename
    with open(target_path, "wb") as f:
        f.write(video.file.read())

    # Compute hash
    sha1 = hashlib.sha1()
    with open(target_path, "rb") as f:
        while True:
            data = f.read(65536)
            if not data:
                break
            sha1.update(data)
    video_hash = sha1.hexdigest()

    dataset_id = video_hash

    # Rename tmp dir with hash string
    dataset_dir = data_dir / dataset_id
    tmp_dir.rename(dataset_dir)
    video_path = dataset_dir / video.filename

    task = worker.process_video.delay(str(video_path))

    curs = conn.cursor()
    curs.execute(
        f"""INSERT INTO datasets (dataset_id, label,
        video_path, video_hash, celery_task_id) VALUES ('{dataset_id}',
        '{label}' , '{str(video_path)}', '{video_hash}', '{task.id}')"""
    )
    curs.close()
    conn.commit()
    return fastapi.responses.JSONResponse({"dataset_id": dataset_id})


@app.get("/trainings/{dataset_id}")
def get_status(dataset_id):
    curs = conn.cursor()
    curs.execute(
        f"""SELECT celery_task_id FROM datasets
        WHERE datasets.dataset_id = '{dataset_id}'"""
    )
    task_id = curs.fetchone()
    curs.close()
    task_result = celery.result.AsyncResult(task_id)
    result = {
        "dataset_id": task_id,
        "status": task_result.status,
        "result": task_result.result,
    }
    return fastapi.responses.JSONResponse(result)
