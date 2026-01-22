from fastapi import FastAPI
from dotenv import load_dotenv
import toml

with open("pyproject.toml", "r") as f:
    config = toml.load(f)
    config: dict = config.get("project")

load_dotenv("./.env")


app = FastAPI(title="FAQChatbot", version=config.get("version"))
