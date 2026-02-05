from dotenv import load_dotenv

from app.model.models import User

load_dotenv("./.env")
PATH_PDF_SAMPLE = "./tests/sample.pdf"
DOCUMENT_METADATA = {"subject": "Test"}
SESSION_ID: str = "1"
USER = User(
    email="test@email.com",
    username="Test",
    password="123",
    full_name="Test Testado",
)
