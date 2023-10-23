import json
from unittest.mock import patch

import pytest
from main import TrainingParams
from worker import train


@pytest.fixture(params=[10, 45])
def max_epochs(request):
    return request.param


@pytest.fixture(params=[5, 20])
def batch_size(request):
    return request.param


# @pytest.fixture
# def make_data():
#     def _make_dataset(n_imgs):
#         return
#
#     yield _make_dataset
#     # Destroy files here
#     # ...


@pytest.fixture
def training_params(max_epochs, batch_size):
    return TrainingParams(
        max_epochs=max_epochs,
        batch_size=batch_size,
        training_data=tuple(),
        validation_data=tuple(),
    )


def test_train(training_params):
    assert train.run(**dict(training_params))


@patch("worker.train.run")
def test_mock_task(mock_run):
    assert train.run(1)
    train.run.assert_called_once_with(1)

    assert train.run(2)
    assert train.run.call_count == 2

    assert train.run(3)
    assert train.run.call_count == 3


def test_task_status(test_app, training_params):
    response = test_app.post("/trainings", data=json.dumps(dict(training_params)))
    content = response.json()
    print(content)
    task_id = content["training_id"]
    assert task_id

    response = test_app.get(f"trainings/{task_id}")
    content = response.json()
    assert content == {
        "training_id": task_id,
        "training_status": "PENDING",
        "training_result": None,
    }
    assert response.status_code == 200

    while content["training_status"] == "PENDING":
        response = test_app.get(f"trainings/{task_id}")
        content = response.json()
    assert content == {
        "training_id": task_id,
        "training_status": "SUCCESS",
        "training_result": True,
    }
