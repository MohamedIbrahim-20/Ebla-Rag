from fastapi import FastAPI
from routes import base, data
from routes.chat import chat_router
from models.db import init_db
app = FastAPI()

@app.on_event("startup")
def on_startup():
    init_db()

app.include_router(base.base_router)

app.include_router(data.data_router)

app.include_router(chat_router)