from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from backend.routes import router
from fastapi.middleware.cors import CORSMiddleware
from backend.config import APP_NAME, VERSION

app = FastAPI(
    title=APP_NAME,
    version=VERSION
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)

app.mount(
    "/uploads",
    StaticFiles(directory="backend/uploads"),
    name="uploads"
)

from backend.services.ocr_services import warmup_reader

@app.on_event("startup")
async def startup():

    warmup_reader()
