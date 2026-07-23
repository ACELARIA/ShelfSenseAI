from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from backend.routes import router
from backend.config import APP_NAME, VERSION

app = FastAPI(
    title=APP_NAME,
    version=VERSION
)

app.include_router(router)

app.mount(
    "/uploads",
    StaticFiles(directory="backend/uploads"),
    name="uploads"
)
