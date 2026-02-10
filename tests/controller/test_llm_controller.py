from typing import Tuple
from tests.controller import client, delete_user, setup_user, insert_user, authenticate
from tests.database.test_database import setup_pdf_sample
from tests import USER


def test_post_new_session(setup_user: Tuple[dict, dict]) -> None:
    message = {"text": "How many hours are in a day?"}
    response = client.post("/llm", json=message)
    assert response.status_code == 200
    response_json = response.json()
    assert response_json["answer"]
    client.delete(
        "/llm/sessions",
        params={"session_id": response_json["session_id"]},
    )


def test_delete_session(setup_user: Tuple[dict, dict]) -> None:
    message = {"text": "How many hours are in a day?"}
    response = client.post("/llm", json=message)
    response_json = response.json()
    response = client.delete(
        "/llm/sessions",
        params={"session_id": response_json["session_id"]},
    )
    assert response.status_code == 200
    response = client.get("/llm/sessions")
    response_json = response.json()
    assert len(response_json["sessions"]) == 0


def test_post_existing_session(setup_user: Tuple[dict, dict]) -> None:
    message = {"text": "How many hours are in a day?"}
    response = client.post("/llm", json=message)
    response_json = response.json()
    message.update(
        {
            "session_id": response_json["session_id"],
            "text": "How many hours are in a year?",
        }
    )
    response = client.post("/llm", json=message)
    assert response.status_code == 200
    assert response.json()["session_id"] == message["session_id"]
    client.delete(
        "/llm/sessions",
        params={"session_id": response_json["session_id"]},
    )


def test_post_session_from_another_user(setup_user: Tuple[dict, dict]) -> None:
    user, _ = setup_user
    message = {"text": "How many hours are in a day?"}
    response_user_1 = client.post("/llm", json=message)
    session_user_1 = response_user_1.json()["session_id"]
    user_2 = {
        "email": "new_test@email.com",
        "username": "New User",
        "password": "321",
        "full_name": "New User 2 ",
    }
    insert_user(user_2)
    authenticate(user_2["email"], user_2["password"])
    message.update({"session_id": session_user_1})
    response_user_2 = client.post("/llm", json=message)
    session_user_2 = response_user_2.json()["session_id"]
    assert session_user_1 != session_user_2
    client.delete(
        "/llm/sessions",
        params={"session_id": session_user_2},
    )
    delete_user()
    authenticate(user["email"], user["password"])
    client.delete(
        "/llm/sessions",
        params={"session_id": session_user_1},
    )


def test_get_sessions(setup_user: Tuple[dict, dict]) -> None:
    response = client.get("/llm/sessions")
    assert response.status_code == 200
    assert "sessions" in response.json().keys()


def test_get_messages_from_session(setup_user: Tuple[dict, dict]) -> None:
    message = {"text": "How many hours are in a day?"}
    response = client.post("/llm", json=message)
    session_id = response.json()["session_id"]
    response = client.get(f"/llm/sessions/{session_id}")
    assert response.status_code == 200
    response_json = response.json()
    assert len(response_json) == 2
    assert message["text"] in [message["content"] for message in response_json]
    client.delete(
        "/llm/sessions",
        params={"session_id": session_id},
    )


def test_post_llm_rag(setup_user: Tuple[dict, dict], setup_pdf_sample: dict) -> None:
    message = {"text": "What is Large Language Models (LLM)?"}
    response = client.post("/llm/rag", json=message)
    assert response.status_code == 200
    response_json = response.json()
    assert response_json["answer"]
    client.delete(
        "/llm/sessions",
        params={"session_id": response_json["session_id"]},
    )
