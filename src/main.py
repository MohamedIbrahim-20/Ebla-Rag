from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from routes import base, data
from routes.chat import chat_router
from models.db import init_db
import os

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")

# Set up templates
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))

@app.on_event("startup")
def on_startup():
    init_db()

# UI route
@app.get("/", response_class=HTMLResponse)
async def chat_ui(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

app.include_router(base.base_router)

app.include_router(data.data_router)

app.include_router(chat_router)