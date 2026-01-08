"""FastAPI main application entry point."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
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


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": config.APP_NAME,
        "version": config.APP_VERSION,
        "status": "running",
        "endpoints": {
            "transform": "/api/v1/transform",
            "validate": "/api/v1/validate",
            "health": "/api/v1/health",
            "scenarios": "/api/v1/scenarios"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level=config.LOG_LEVEL.lower())
