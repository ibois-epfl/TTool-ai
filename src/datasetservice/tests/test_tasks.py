import pathlib
import random
from unittest.mock import patch

import cv2
import numpy as np
import pytest
from worker import process_video

random.seed(42)


@pytest.fixture
def video_file(tmp_path):
    video_path = tmp_path / "test_video.mp4"
    fps = 29.974512
    duration = 5
    size = (720, 1280)
    out = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size[::-1], False
    )
    for _ in range(int(fps * duration)):
        data = np.random.randint(0, 256, size, dtype="uint8")
        out.write(data)
    out.release()
    return video_path


def test_process_video(video_file):
    process_video.run(video_file)
    directory = video_file.parents[0]
    train_dir = directory / "train"
    val_dir = directory / "val"
    assert train_dir.is_dir()
    assert val_dir.is_dir()
    n_train = len(list(train_dir.glob("*.png")))
    n_val = len(list(val_dir.glob("*.png")))
    print(n_train, n_val)
    assert np.isclose(n_train / (n_val + n_train), 0.8, atol=0.05)


@patch("worker.process_video.run")
def test_mock_task(mock_run):
    path = pathlib.Path("vid1.mp4")
    assert process_video.run(path)
    process_video.run.assert_called_once_with(path)

    path = pathlib.Path("vid2.mp4")
    assert process_video.run(path)
    assert process_video.run.call_count == 2

    path = pathlib.Path("vid3.mp4")
    assert process_video.run(path)
    assert process_video.run.call_count == 3


def test_task_status(test_app, video_file):
    file_name = video_file.stem
    with open(video_file, "rb") as f:
        response = test_app.post(
            "/datasets",
            params={"label": "test_label"},
            files={"video": (file_name, f, "video/mp4")},
        )
    content = response.json()
    dataset_id = content["dataset_id"]
    assert dataset_id

    response = test_app.get(f"datasets/{dataset_id}")
    content = response.json()
    assert content == {
        "dataset_id": dataset_id,
        "status": "PENDING",
        "result": None,
    }
    assert response.status_code == 200

    while content["status"] == "PENDING":
        response = test_app.get(f"dataset/{dataset_id}")
        content = response.json()
    assert content == {
        "dataset_id": dataset_id,
        "status": "SUCCESS",
        "result": True,
    }
