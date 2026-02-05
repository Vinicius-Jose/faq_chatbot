from typing import Generator
import pytest
from tests.controller import client, setup_user
from tests import USER


@pytest.mark.parametrize(
    "email, pwd, status_code",
    [
        (USER.email, USER.password, 200),
        ("master_chief@email.com", "halo", 401),
    ],
)
def test_token(setup_user: dict, email: str, pwd: str, status_code: int) -> None:
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    payload = f"grant_type=password&username={email}&password={pwd}"
    response = client.post("/user/token", data=payload, headers=headers)
    assert response.status_code == status_code
    if status_code == 200:
        response = response.json()
        assert response["access_token"]
        assert response["token_type"] == "Bearer"
