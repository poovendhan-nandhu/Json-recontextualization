"""FastAPI main application entry point."""
import logging
import sys
import os
from pathlib import Path

# Configure logging FIRST - before any other imports
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from src.api.routes import router as api_router
from src.utils.config import config

# Create FastAPI app
app = FastAPI(
    title=config.APP_NAME,
    version=config.APP_VERSION,
    description="Scenario-Aware JSON Re-Contextualization API using LangGraph and OpenAI"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router, prefix="/api/v1", tags=["transformation"])

# Serve static frontend files
STATIC_DIR = Path(__file__).parent.parent / "static"
if STATIC_DIR.exists():
    # Mount static files for assets
    app.mount("/_next", StaticFiles(directory=STATIC_DIR / "_next"), name="next_static")

    @app.get("/", response_class=HTMLResponse)
    async def serve_frontend():
        """Serve the frontend app."""
        index_file = STATIC_DIR / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        return HTMLResponse("<h1>Frontend not built. Run: cd frontend && npm run build</h1>")

    @app.get("/{path:path}")
    async def serve_static(path: str, request: Request):
        """Serve static files or fallback to index.html for SPA routing."""
        # Skip API routes
        if path.startswith("api/"):
            return None

        file_path = STATIC_DIR / path

        # Try exact file
        if file_path.is_file():
            return FileResponse(file_path)

        # Try with .html extension
        html_path = STATIC_DIR / f"{path}.html"
        if html_path.is_file():
            return FileResponse(html_path)

        # Try index.html in directory
        index_path = file_path / "index.html"
        if index_path.is_file():
            return FileResponse(index_path)

        # Fallback to main index.html for SPA
        return FileResponse(STATIC_DIR / "index.html")
else:
    @app.get("/")
    async def root():
        """Root endpoint - frontend not built."""
        return {
            "name": config.APP_NAME,
            "version": config.APP_VERSION,
            "status": "running",
            "frontend": "not built - run: cd frontend && npm run build",
            "api_docs": "/docs",
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level=config.LOG_LEVEL.lower())
