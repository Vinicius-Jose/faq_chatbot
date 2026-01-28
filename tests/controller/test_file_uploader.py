from fastapi.testclient import TestClient
from app.main import app
from tests import PATH_PDF_SAMPLE, DOCUMENT_METADATA


client = TestClient(app)


def test_post_file() -> None:
    pdf_file = open(PATH_PDF_SAMPLE, "rb")
    files = {"file": (pdf_file.name, pdf_file.read(), "application/pdf")}
    response = client.post(
        "/files/",
        data={"document_subject": DOCUMENT_METADATA.get("subject")},
        files=files,
    )
    assert response.status_code == 200


def test_post_file_not_supported() -> None:
    file = open("./.env", "rb")
    files = {"file": (file.name, file.read(), "text/plain")}
    response = client.post(
        "/files/",
        data={"document_subject": DOCUMENT_METADATA.get("subject")},
        files=files,
    )
    assert response.status_code == 400


def test_delete_file_with_subject() -> None:
    response = client.delete(
        f"/files/{DOCUMENT_METADATA.get('subject')}",
    )
    assert response.status_code == 200
