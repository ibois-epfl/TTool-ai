from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import config.constants as constants
import os

app = FastAPI()

@app.post("/upload_videos", status_code=201)
async def upload_videos(file: UploadFile):
    # save the video to a folder
    target_path = os.path.join(constants.SAVE_VIDEO_PATH, file.filename)
    with open(target_path, "wb") as video_file:
        video_file.write(file.file.read())


    # write the video to a database


    # produce the video to the queue for the dataloader

    return JSONResponse(content={"message": "Video uploaded successfully"}, status_code=200)


@app.post("/set_train_config", status_code=201)
def set_train_config():

    return {"message": "Training configuration set successfully"}
