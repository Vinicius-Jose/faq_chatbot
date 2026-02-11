FROM python:3.10.18-slim-bullseye

WORKDIR /app

COPY ./app /app

RUN pip install poetry \
    && poetry install --without dev \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

EXPOSE 8000

CMD ["poetry", "run","uvicorn", "app.main:app", "--host", "0.0.0.0","--port", "8000","--reload"]