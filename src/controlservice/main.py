from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, FileResponse
import os
import pika
import hashlib
from fastapi import UploadFile, File
from fastapi.encoders import jsonable_encoder
from datetime import datetime
import json
from sqlalchemy import func
import uuid
import re
import shutil
import zipfile
from tempfile import NamedTemporaryFile
from models import VideoDB, Status, TrainDB
from models import TrainConfig
from config import SessionLocal, init_db
from config import producer_rabbit_channel, RABBITMQ_DATA_QUEUE, RABBITMQ_TRAIN_QUEUE
from config import DOCKER_VIDEO_DIR

app = FastAPI()

@app.on_event("startup")
def on_startup():
    init_db()

def standard_label(input_label):
    standard_label = input_label.lower()
    standard_label = re.sub(r"[^\w\s]", " ", standard_label)
    standard_label = re.sub(r"\s+", " ", standard_label)
    standard_label = standard_label.strip()
    standard_label = standard_label.replace(" ", "_")
    return standard_label

@app.post("/upload_videos", status_code=201)
async def upload_videos(video: UploadFile = File(...), label: str = File(...)):
    db_session = SessionLocal()
    try:
        try:
            label = standard_label(label)
            timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
            video_name = f"{label}_{timestamp}.{video.filename.split('.')[-1]}"
            label_dir = os.path.join(DOCKER_VIDEO_DIR, label, video_name.split(".")[0])
            os.makedirs(label_dir, exist_ok=True)
        except Exception as e:
            return JSONResponse(content={"message": "Error while creating video directory"},
                                status_code=500)

        target_path = os.path.join(label_dir, video_name)
        with open(target_path, "wb") as video_file:
            content = await video.read()
            video_file.write(content)
    except Exception as e:
        return JSONResponse(content={"message": "Error while saving video"},
                            status_code=500)

    video_hash = hashlib.sha256(content).hexdigest()
    try:
        try:
            new_video = VideoDB(
                label=label,
                video_path=target_path,
                video_hash=video_hash,
                data_dir=label_dir
            )
            db_session.add(new_video)
            db_session.commit()
        except Exception as e:
            db_session.rollback()
            if label_dir and os.path.exists(label_dir):
                shutil.rmtree(label_dir)
            return JSONResponse(content={"message": "This video is already exists!"},
                                status_code=409)

        try:
            video_json = jsonable_encoder(new_video.video_path)
            producer_rabbit_channel.basic_publish(exchange='',
                                                  routing_key=RABBITMQ_DATA_QUEUE,
                                                  body=video_json,
                                                  properties=pika.BasicProperties(delivery_mode=2))
        except Exception as e:
            return JSONResponse(content={"message": "Error while sending video for further processing"},
                                status_code=500)

        video_id = new_video.id
        return JSONResponse(content={"VIDEO_ID": video_id,
                                     "message": "Video uploaded successfully. "
                                                "Use the provided VIDEO_ID to check status or delete the video."},
                            status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail="An unexpected error occurred. Please try again.")
    finally:
        db_session.close()


@app.get("/check_upload_video_status/{video_id}", status_code=200)
def check_upload_video_status(video_id: int):
    db_session = SessionLocal()
    try:
        query = db_session.query(VideoDB).filter(VideoDB.id == video_id).first()
        if not query:
            return JSONResponse(content={"message": "Video id not found in database."}, status_code=404)
        if query.upload_status == Status.COMPLETED:
            return JSONResponse(content={"message": "Video uploaded successfully."}, status_code=200)
        elif query.upload_status == Status.FAILED:
            return JSONResponse(content={"message": "Video upload failed. Please try again."}, status_code=500)
        elif query.upload_status == Status.PROCESSING:
            return JSONResponse(content={"message": "Video is being processed. Please wait."}, status_code=202)
        else:
            return JSONResponse(content={"message": "Video upload pending. Please wait."}, status_code=202)
    except Exception as e:
        raise HTTPException(status_code=500, detail="An unexpected error occurred. Please try again.")
    finally:
        db_session.close()


@app.delete("/delete_video/{video_id}", status_code=200)
def delete_video(video_id: int):
    db_session = SessionLocal()
    try:
        query = db_session.query(VideoDB).filter(VideoDB.id == video_id).first()
        if not query:
            return JSONResponse(content={"message": "Video id not found in database."}, status_code=404)
        if query.upload_status == Status.COMPLETED or query.upload_status == Status.FAILED:
            if os.path.exists(query.data_dir):
                shutil.rmtree(query.data_dir)
            db_session.delete(query)
            db_session.commit()
            return JSONResponse(content={"message": "Video deleted successfully."}, status_code=200)
        else:
            return JSONResponse(content={"message": "Video is being processed. Please wait. "
                                                    "You can delete the video only when the status is completed"},
                                status_code=202)
    except Exception as e:
        raise HTTPException(status_code=500, detail="An unexpected error occurred. Please try again.")
    finally:
        db_session.close()


@app.get("/get_classes_available", status_code=200)
def get_classes_available():
    db_session = SessionLocal()
    try:
        labels = db_session.query(VideoDB.label).filter(VideoDB.upload_status == Status.COMPLETED).distinct().all()
        labels = [label[0] for label in labels]
    except Exception as e:
        return JSONResponse(content={"message": "Error while getting classes from database"}, status_code=500)
    finally:
        db_session.close()
    return sorted(labels)

@app.post("/train_model", status_code=201)
def train_model(config: TrainConfig):
    db_session = SessionLocal()
    try:
        label_dirs = []
        results = (db_session.query
                   (VideoDB.label, func.array_agg(VideoDB.data_dir), )
                   .filter(VideoDB.label.in_(config.classes)).group_by(VideoDB.label).all())
        for l, dir in results:
            label_dirs.append(dir)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Set Config Error")

    try:
        if not label_dirs:
            return JSONResponse(content={"message": "No data for training."},
                                status_code=404)
        if config.epochs <= 0:
            return JSONResponse(content={"message": "Max epochs must be positive."},
                                status_code=400)
        if config.batch_size <= 0:
            return JSONResponse(content={"message": "Batch size must be positive."},
                                status_code=400)
        user_id = uuid.uuid4().hex
        train_config = {
            "user_id": user_id,
            "data_dirs": label_dirs,
            "max_epochs": config.epochs,
            "batch_size": config.batch_size,
            "classes": config.classes
        }
        message = json.dumps(train_config)
        print("Sending message to rabbitmq queue: ", train_config)
        producer_rabbit_channel.basic_publish(exchange='',
                                              routing_key=RABBITMQ_TRAIN_QUEUE,
                                              body=message,
                                              properties=pika.BasicProperties(delivery_mode=2))
    except Exception as e:
        return JSONResponse(content={"message": "Error while starting the training. Please try again."},
                            status_code=500)
    return JSONResponse(content={"message": f"Training has started successfully. USER_ID: {user_id}. " 
                                            "Please save the USER_ID to check the status of the training "
                                            "and get the train model"},
                        status_code=200)


@app.get("/check_train_status/{user_id}", status_code=200)
def check_train_status(user_id: str):
    db_session = SessionLocal()
    try:
        query = db_session.query(TrainDB).filter(TrainDB.user_id == user_id).first()
        if query:
            train_status = query.status
        else:
            return JSONResponse(content={"message": "User id not found in database."}, status_code=404)
    except Exception as e:
        return JSONResponse(content={"message": "Error while getting train status from database."}, status_code=500)
    finally:
        db_session.close()

    if train_status == Status.COMPLETED:
        return JSONResponse(content={"message": "Train completed successfully."}, status_code=200)
    elif train_status == Status.FAILED:
        return JSONResponse(content={"message": "Train failed. Please try again."}, status_code=500)
    elif train_status == Status.TRAINING:
        return JSONResponse(content={"message": "Train is being processed. Please wait."}, status_code=202)
    else:
        return JSONResponse(content={"message": "Train pending. Please wait."}, status_code=202)


@app.get("/get_train_model/{user_id}", status_code=200)
def get_train_model(user_id: str):
    db_session = SessionLocal()
    try:
        query = db_session.query(TrainDB).filter(TrainDB.user_id == user_id).first()
        if not query:
            return JSONResponse(content={"message": "User id not found in database."}, status_code=404)
        if query.status == Status.COMPLETED:
            ml_model = query.trace_file
            label_map_file = query.label_map_file
            if ml_model and label_map_file and os.path.isfile(ml_model) and os.path.isfile(label_map_file):
                with NamedTemporaryFile(delete=False, suffix=".zip") as temp_zip:
                    with zipfile.ZipFile(temp_zip, 'w') as zipf:
                        zipf.write(ml_model, "efficientnet.pt")
                        zipf.write(label_map_file, os.path.basename(label_map_file))
                    temp_zip_path = temp_zip.name
                return FileResponse(temp_zip_path, media_type="application/zip", filename="trained_model_and_labels.zip")
            else:
                return JSONResponse(content={"message": "Model or Label Map file not found in database."}, status_code=404)
        else:
            return JSONResponse(content={"message": "Train not completed yet. Please wait."}, status_code=404)
    except Exception as e:
        raise HTTPException(status_code=500, detail="An unexpected error occurred. Please try again.")



