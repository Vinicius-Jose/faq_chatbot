from tests.controller import client


def test_post() -> None:
    message = {"user_id": "1", "text": "How many hours does a day have?"}
    response = client.post("/llm", json=message)
    assert response.status_code == 200
    assert response.json()["answer"]
