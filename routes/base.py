from fastapi import FastAPI, APIRouter

base_router = APIRouter()

@base_router.get("/")
def welcome_message():
    return {"message": "Welcome to the Rag application!"}