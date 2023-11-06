from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from models.control_service_api_model import VideoUploadApiRequest
from models.postgres_model import VideoDB, VideoStatus
from config.postgres_config import SessionLocal, init_db
import config.constants as constants
import os
import hashlib
from fastapi import UploadFile, File

app = FastAPI()


@app.on_event("startup")
def on_startup():
    init_db()


@app.post("/upload_videos", status_code=201)
async def upload_videos(video: UploadFile = File(...), label: str = File(...)):
    db_session = SessionLocal()
    # save the video to a folder
    try:
        target_path = os.path.join(constants.SAVE_VIDEO_PATH, video.filename)
        with open(target_path, "wb") as video_file:
            content = await video.read()
            video_file.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    video_hash = hashlib.sha256(content).hexdigest()
    try:
        new_video = VideoDB(
                label=label,
                video_path=target_path,
                video_hash=video_hash,
                upload_status=VideoStatus.PROCESSING
            )
        db_session.add(new_video)
        db_session.commit()

        new_video.upload_status = VideoStatus.COMPLETED
        db_session.commit()

        video_id = new_video.id
        return JSONResponse(content={"video_id": video_id,
                                     "status": VideoStatus.COMPLETED,
                                     "message": "Video uploaded successfully."},
                            status_code=200)
    except Exception as e:
        db_session.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db_session.close()


@app.post("/set_train_config", status_code=201)
def set_train_config():

    return {"message": "Training configuration set successfully"}
