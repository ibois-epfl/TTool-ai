from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from models.postgres_model import VideoDB, DataLoaderStatus, TrainDB
from models.train_config_api import TrainConfig
from config.postgres_config import SessionLocal, init_db
from config.rabbit_config import producer_rabbit_channel, RABBITMQ_DATA_QUEUE, RABBITMQ_TRAIN_QUEUE
import config.constants as constants
import os
import pika
import hashlib
from fastapi import UploadFile, File
from sqlalchemy.exc import IntegrityError
from fastapi.encoders import jsonable_encoder
from datetime import datetime
import json
from sqlalchemy import func

app = FastAPI()

@app.on_event("startup")
def on_startup():
    init_db()

@app.post("/upload_videos", status_code=201)
async def upload_videos(video: UploadFile = File(...), label: str = File(...)):
    db_session = SessionLocal()
    try:
        try:
            timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            video_name = f"{label}-{timestamp}.{video.filename.split('.')[-1]}"
            label_dir = os.path.join(constants.DOCKER_VIDEO_DIR, label, video_name.split(".")[0])
            os.makedirs(label_dir, exist_ok=True)
        except Exception as e:
            raise HTTPException(status_code=500, detail="Error while creating video directory")

        target_path = os.path.join(label_dir, video_name)
        with open(target_path, "wb") as video_file:
            content = await video.read()
            video_file.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error while saving video")

    video_hash = hashlib.sha256(content).hexdigest()
    try:
        try:
            new_video = VideoDB(
                    label=label,
                    video_path=os.path.join(label_dir, video_name),
                    video_hash=video_hash,
                    data_dir=label_dir
                )
            db_session.add(new_video)
            db_session.commit()
        except Exception as e:
            db_session.rollback()
            raise HTTPException(status_code=500, detail="Error while adding video to database")
        except IntegrityError:
            db_session.rollback()
            print("A video with the same hash already exists.")

        try:
            video_json = jsonable_encoder(new_video.video_path)
            producer_rabbit_channel.basic_publish(exchange='',
                                                  routing_key=RABBITMQ_DATA_QUEUE,
                                                  body=video_json,
                                                  properties=pika.BasicProperties(delivery_mode=2))
        except Exception as e:
            print("Error while sending message to rabbitmq queue: ", str(e))

        video_id = new_video.id
        return JSONResponse(content={"video_id": video_id,
                                     "message": "Video uploaded successfully."},
                            status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db_session.close()

@app.get("/get_classes_available", status_code=200)
def get_classes_available():
    db_session = SessionLocal()
    try:
        labels = db_session.query(VideoDB.label).filter(VideoDB.upload_status == DataLoaderStatus.COMPLETED).distinct().all()
        labels = [label[0] for label in labels]
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error while getting labels from database.")
    finally:
        db_session.close()
    return labels


@app.post("/set_train_config", status_code=201)
def set_train_config(config: TrainConfig):
    db_session = SessionLocal()
    try:
        label_dirs = []
        results = (db_session.query
                   (VideoDB.label,func.array_agg(VideoDB.data_dir),)
                   .filter(VideoDB.label.in_(config.classes)).group_by(VideoDB.label).all())
        for l, dir in results:
            label_dirs.append(dir)
    except Exception as e:
        print("Error while getting label directories from database: ", str(e))
        raise HTTPException(status_code=500, detail="Set Config Error")
    try:
        new_train = TrainDB(labels=config.classes)
        db_session.add(new_train)
        db_session.commit()
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error while adding train config to database")

    try:
        user_id = jsonable_encoder(new_train.id)
        train_config = {
            "user_id": user_id,
            "data_dirs": label_dirs,
            "max_epochs": config.epochs,
            "batch_size": config.batch_size,
        }
        message = json.dumps(train_config)
        print("Sending message to rabbitmq queue: ", train_config)
        producer_rabbit_channel.basic_publish(exchange='',
                                              routing_key=RABBITMQ_TRAIN_QUEUE,
                                              body=message,
                                              properties=pika.BasicProperties(delivery_mode=2))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error while sending message to rabbitmq queue")

    return JSONResponse(content={"message": "Train config set successfully. YOUR USER ID: " + str(user_id)},
                        status_code=200)


@app.get("/get_train_status/{user_id}", status_code=200)
def get_train_status(user_id: int):
    db_session = SessionLocal()
    try:
        query = db_session.query(TrainDB).filter(TrainDB.id == user_id).first()
        if query:
            train_status = query.train_status
        else:
            raise HTTPException(status_code=404, detail="User id not found in database.")
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error while getting train status from database.")
    finally:
        db_session.close()
    return train_status

@app.get("/get_train_result/{user_id}", status_code=200)
def get_train_result(user_id: int):
    db_session = SessionLocal()
    try:
        query = db_session.query(TrainDB).filter(TrainDB.id == user_id).first()
        if not query:
            raise HTTPException(status_code=404, detail="User id not found in database.")
        if query.train_status == DataLoaderStatus.COMPLETED:
            model_path = query.model_path
            if model_path and os.path.isfile(model_path):
                return FileResponse(model_path, media_type="application/octet-stream", filename="ac_model.pt")
            else:
                raise HTTPException(status_code=404, detail="Model not found in database.")
        else:
            raise HTTPException(status_code=404, detail="Train not completed yet.")
    except Exception as e:
        raise HTTPException(status_code=500, detail="Get Train Result API Error.")
