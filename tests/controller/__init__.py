from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app, headers={"Content-Type": "application/json"})
