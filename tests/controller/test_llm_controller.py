from tests.controller import client
from tests import USER


def test_post() -> None:
    message = {"user_email": USER.email, "text": "How many hours does a day have?"}
    response = client.post("/llm", json=message)
    assert response.status_code == 200
    assert response.json()["answer"]
