from fastapi import Depends, FastAPI
from dotenv import load_dotenv

load_dotenv("./.env")
from app.controller.file_uploader import router as file_route
from app.controller.llm import router as llm_router
from app.controller.user import router as user_router, get_current_active_user
import toml


with open("pyproject.toml", "r") as f:
    config = toml.load(f)
    config: dict = config.get("project")


app = FastAPI(title="FAQChatbot", version=config.get("version"))
app.include_router(file_route, dependencies=[Depends(get_current_active_user)])
app.include_router(llm_router, dependencies=[Depends(get_current_active_user)])
app.include_router(user_router)
