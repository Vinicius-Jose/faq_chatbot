from typing import Generator, Tuple
from fastapi.testclient import TestClient
import pytest
from app.main import app
from app.model.models import User
from tests import USER

client = TestClient(app, headers={"Content-Type": "application/json"})


def authenticate(email: str, pwd: str) -> dict:
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    payload = f"grant_type=password&username={email}&password={pwd}"
    response = client.post("/user/token", data=payload, headers=headers)
    response = response.json()
    headers = {"Authorization": f"{response['token_type']} {response['access_token']}"}
    client.headers.update(headers)
    return headers


def insert_user(user: dict):
    client.post("/user/", json=user)


@pytest.fixture(scope="module")
def setup_user() -> Generator[Tuple[dict, dict], None, None]:
    user = {
        "email": USER.email,
        "username": USER.username,
        "password": USER.password,
        "full_name": USER.full_name,
    }
    insert_user(user)
    headers = authenticate(USER.email, USER.password)
    yield user, headers
    delete_user()


def delete_user() -> None:
    client.delete("/user/")
