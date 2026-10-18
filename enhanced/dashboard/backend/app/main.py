"""Main FastAPI application."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pathlib import Path

from app.database import init_db
from app.api import predictions, explanations, metrics, patients
from app.services.xai_loader import XAI_DIR

# Initialize database on startup
init_db()

app = FastAPI(
    title="Sepsis Prediction Dashboard API",
    description="API for sepsis early warning system with explainability",
    version="1.0.0"
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(predictions.router)
app.include_router(explanations.router)
app.include_router(metrics.router)
app.include_router(patients.router)


@app.get("/")
def root():
    return {
        "name": "Sepsis Prediction Dashboard API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "ok"
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/static/xai/{filename}")
def serve_xai_image(filename: str):
    file_path = XAI_DIR / filename
    if file_path.exists():
        return FileResponse(file_path)
    from fastapi import HTTPException
    raise HTTPException(status_code=404, detail="File not found")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)