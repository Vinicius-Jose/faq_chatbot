from datetime import datetime, timedelta, timezone
from os import getenv
from typing import Annotated
from fastapi import Depends, HTTPException, status, APIRouter
from jwt.exceptions import InvalidTokenError
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
import jwt
from passlib.context import CryptContext


from app.database.database import Neo4jDatabase
from app.model.models import Token, TokenData, User
from app.services.llm import EmbbeddingHuggingFace

router = APIRouter(prefix="/user", tags=["user"])

SECRET_KEY = getenv("SECRET_KEY")
ALGORITHM = getenv("ALGORITHM")
ACCESS_TOKEN_EXPIRE_MINUTES = int(getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 15))

pwd_context = CryptContext(schemes=["bcrypt_sha256"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    return pwd_context.hash(password)


def get_user(email: str):
    embedder = EmbbeddingHuggingFace()
    db = Neo4jDatabase(embedder)
    try:
        user: User = db.get_basemodel(User(email=email))
    except Exception:
        return None
    return user


def authenticate_user(email: str, password: str):
    user = get_user(email)
    if not user:
        return False
    if not verify_password(password, user.password):
        return False
    return user


def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(
            minutes=ACCESS_TOKEN_EXPIRE_MINUTES
        )
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)],
):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except InvalidTokenError:
        raise credentials_exception
    user = get_user(email=token_data.username)
    if user is None:
        raise credentials_exception
    return user


async def get_current_active_user(
    current_user: Annotated[User, Depends(get_current_user)],
):
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Inactive user"
        )
    return current_user


@router.post("/token")
async def login_for_access_token(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()]
) -> Token:
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    return Token(access_token=access_token, token_type="Bearer")


@router.get("/me/", response_model=User, tags=["user"])
async def read_users_me(
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    return current_user


@router.post("/")
def create_user(user: User) -> User:
    user.password = get_password_hash(user.password)
    embedder = EmbbeddingHuggingFace()
    db = Neo4jDatabase(embedder)
    try:
        db.get_basemodel(user)
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "User already exists")
    except (IndexError, TypeError):
        db.save_basemodel(user)
    return user


@router.delete(
    "/",
    dependencies=[Depends(get_current_active_user)],
    tags=["user"],
)
def delete_current_user(
    current_user: Annotated[User, Depends(get_current_active_user)]
) -> User:
    embedder = EmbbeddingHuggingFace()
    db = Neo4jDatabase(embedder)
    db.delete_basemodel(current_user)

    return current_user
