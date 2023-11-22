import json
import pathlib
from unittest.mock import patch

import sqlalchemy
import training
from training_worker import Base, Callback, TrainingDB, TrainingParams


@patch("training_worker.Base.metadata.create_all", spec=Base.metadata.create_all)
@patch("training_worker.TrainingParams", spec=TrainingParams)
@patch("training_worker.TrainingDB", spec=TrainingDB)
@patch("sqlalchemy.create_engine", spec=sqlalchemy.create_engine)
@patch("sqlalchemy.orm.Session", spec=sqlalchemy.orm.Session)
@patch("training.train", spec=training.train)
def test_Callback(
    mock_train,
    mock_session,
    mock_create_engine,
    mock_training_db,
    mock_training_params,
    mock_create_all,
):
    # Setup
    train_params = {
        "max_epochs": 45,
        "batch_size": 20,
        "data_dirs": ("/data_dir1", "/data_dir2"),
    }
    mock_training_params.return_value.max_epochs = train_params["max_epochs"]
    mock_training_params.return_value.batch_size = train_params["batch_size"]
    mock_training_params.return_value.data_dirs = train_params["data_dirs"]
    mock_training_params.return_value.__hash__.return_value = 12345
    train_hash = hash(mock_training_params())
    mock_training_params.reset_mock()

    body = json.dumps(train_params)
    body = body.encode("utf-8")

    # Run
    callback = Callback()
    callback.connect("URL")
    callback.callback(None, None, None, body)

    # Assertions
    mock_create_all.assert_called_once_with(mock_create_engine.return_value)
    mock_training_params.assert_called_once_with(body)

    mock_training_db.assert_called_once_with(
        max_epochs=train_params["max_epochs"],
        data_dirs=train_params["data_dirs"],
        training_hash=train_hash,
    )

    log_dir = pathlib.Path(f"/data/{train_hash}")
    assert log_dir.is_dir()

    mock_train.assert_called_once_with(**train_params, log_dir=log_dir)

    mock_create_engine.assert_called_once_with("URL")
    mock_session.assert_called_with(mock_create_engine.return_value)
