from fastapi import FastAPI

app = FastAPI(
    title="ShelfSense AI",
    version="1.0.0"
)

@app.get("/")
def root():
    return {
        "message": "ShelfSense AI Backend Running"
    }

@app.get("/health")
def health():
    return {
        "status": "healthy"
    }
