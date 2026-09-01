"""
FastAPI Main Application Entrypoint for Career AI REST Service.
Provides interactive OpenAPI/Swagger docs at /docs and ReDoc at /redoc.
"""

import os
import sys
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from api.routes import router

app = FastAPI(
    title="CareerAI REST API - Career Prediction & Skill Gap Analysis",
    description="""
## AI-Powered Career Prediction, Recommendation & Skill Gap Analysis REST Service

Provides high-performance, validated REST endpoints for:
- **Prediction**: Career classification via trained ML ensemble models.
- **Recommendation**: Multi-modal Top-K ranked career matching with ML probability, skill alignment, and Sentence-BERT semantic similarity.
- **Skill Gap Analysis**: Automated detection of matched/missing technical competencies with actionable learning roadmaps.
    """,
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Enable CORS for flexible integration (e.g., Streamlit, React frontends)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Custom request validation error handler providing clear error messages with HTTP 422."""
    error_details = []
    for err in exc.errors():
        loc = " -> ".join(str(l) for l in err.get("loc", []))
        msg = err.get("msg", "Validation error")
        error_details.append(f"{loc}: {msg}")

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "status": "error",
            "error_type": "ValidationError",
            "message": "Invalid request payload provided.",
            "details": error_details
        }
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Global exception handler capturing unexpected internal server errors with HTTP 500."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "status": "error",
            "error_type": type(exc).__name__,
            "message": str(exc)
        }
    )


if __name__ == "__main__":
    import uvicorn
    host = os.environ.get("API_HOST", "127.0.0.1")
    port = int(os.environ.get("API_PORT", 8000))
    print(f"\n[FastAPI] Starting Uvicorn server at http://{host}:{port}")
    print(f"[FastAPI] Swagger Documentation available at http://{host}:{port}/docs")
    print(f"[FastAPI] ReDoc Documentation available at http://{host}:{port}/redoc\n")
    uvicorn.run("api.main:app", host=host, port=port, reload=True)
