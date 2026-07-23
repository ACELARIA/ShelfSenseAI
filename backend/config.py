from pathlib import Path

APP_NAME = "ShelfSense AI"

VERSION = "1.0.0"

UPLOAD_DIR = Path("backend/uploads")

UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True
)

ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
}

MAX_FILE_SIZE = 10 * 1024 * 1024