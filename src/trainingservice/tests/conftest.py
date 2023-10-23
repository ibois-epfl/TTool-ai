import pytest
from starlette.testclient import TestClient

from main import app


@pytest.fixture(scope="module")
def test_app():
    """
    Yields a client that can be used to
    send http requests to the application.
    """
    client = TestClient(app)
    yield client
