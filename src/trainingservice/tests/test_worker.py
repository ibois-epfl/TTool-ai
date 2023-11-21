import json
from unittest.mock import patch

import sqlalchemy
import training
import training_worker


@patch("sqlalchemy.create_engine", spec=sqlalchemy.create_engine)
@patch("sqlalchemy.orm.Session", spec=sqlalchemy.orm.Session)
@patch("training.train", spec=training.train)
def test_Callback(mock_train, mock_session, mock_create_engine):
    callback = training_worker.Callback()
    callback.connect("URL")
    train_params = {
        "data_dirs": ["/data_dir1", "/data_dir2"],
        "max_epochs": 45,
        "batch_size": 20,
        "log_dir": "/logs",
    }
    body = json.dumps(train_params)
    body = body.encode("utf-8")
    callback.callback(None, None, None, body)
    mock_train.assert_called_once_with(**train_params)
    mock_create_engine.assert_called_once_with("URL")
    mock_session.assert_called_with(mock_create_engine.return_value)
