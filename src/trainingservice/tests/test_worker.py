from unittest.mock import Mock, patch

import sqlalchemy
import training
import training_worker


@patch("sqlalchemy.create_engine", spec=sqlalchemy.create_engine)
@patch("sqlalchemy.orm.Session", spec=sqlalchemy.orm.Session)
@patch("training.train", spec=training.train)
def test_Callback(mock_train, mock_session, mock_create_engine):
    callback = training_worker.Callback()
    callback.connect("URL")
    body = {"datasets": ["12345", "67891"], }
    callback.callback(None, None, None, body)
    mock_train.assert_called_once_with()
    mock_create_engine.assert_called_once_with("URL")
    mock_session.assert_called_with(mock_create_engine.return_vlaue)
