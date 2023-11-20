from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from models.postgres_model import VideoDB, DataLoaderStatus
from config.postgres_config import SessionLocal, init_db
from config.rabbit_config import producer_rabbit_channel, RABBITMQ_DATA_QUEUE
import config.constants as constants
import os
import pika
import hashlib
from fastapi import UploadFile, File
from fastapi.encoders import jsonable_encoder
import json

app = FastAPI()


@app.on_event("startup")
def on_startup():
    init_db()


@app.post("/upload_videos", status_code=201)
async def upload_videos(video: UploadFile = File(...), label: str = File(...)):
    db_session = SessionLocal()
    try:
        target_path = os.path.join(constants.SAVE_VIDEO_PATH, video.filename)
        with open(target_path, "wb") as video_file:
            content = await video.read()
            video_file.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    video_hash = hashlib.sha256(content).hexdigest()
    try:
        try:
            new_video = VideoDB(
                    label=label,
                    video_path=target_path,
                    video_hash=video_hash,
                    upload_status=DataLoaderStatus.PENDING
                )
            db_session.add(new_video)
            db_session.commit()
        except Exception as e:
            db_session.rollback()
            raise HTTPException(status_code=500, detail=str(e))

        try:
            video_data = jsonable_encoder(new_video)
            video_json = json.dumps(video_data)
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

@app.post("/set_train_config", status_code=201)
def set_train_config():

    return {"message": "Training configuration set successfully"}
