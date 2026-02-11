from typing import Tuple
from tests.controller import client, setup_user
from tests import PATH_PDF_SAMPLE, DOCUMENT_METADATA


def test_post_file(setup_user: Tuple[dict, dict]) -> None:
    _, _ = setup_user
    pdf_file = open(PATH_PDF_SAMPLE, "rb")
    files = {"file": (pdf_file.name, pdf_file.read(), "application/pdf")}
    response = client.post(
        "/files/",
        data={
            "document_subject": DOCUMENT_METADATA.get("subject"),
            "file_name": DOCUMENT_METADATA.get("file_name"),
        },
        files=files,
    )
    assert response.status_code == 200


def test_post_file_not_supported(setup_user: Tuple[dict, dict]) -> None:
    _, _ = setup_user
    file = open("./.env", "rb")
    files = {
        "file": (file.name, file.read(), "text/plain"),
    }
    response = client.post(
        "/files/",
        files=files,
        data={
            "document_subject": DOCUMENT_METADATA.get("subject"),
            "file_name": DOCUMENT_METADATA.get("file_name"),
        },
    )
    assert response.status_code == 400


def test_delete_file_with_subject(setup_user: Tuple[dict, dict]) -> None:
    _, _ = setup_user
    response = client.delete(
        f"/files/{DOCUMENT_METADATA.get('subject')}",
    )
    assert response.status_code == 200
