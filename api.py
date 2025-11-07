"""
Clinical Trial RAG Chatbot - REST API Service
FastAPI-based API service for the chatbot
"""
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from contextlib import asynccontextmanager
import time
from chatbot import RAGChatbot
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global chatbot instance
chatbot = None


# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    # Startup
    global chatbot
    try:
        logger.info("Loading chatbot...")
        chatbot = RAGChatbot()
        logger.info("Chatbot loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load chatbot: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down API service")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Clinical Trial RAG Chatbot API",
    description="API service for querying clinical trial information using RAG technology",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models
class ChatRequest(BaseModel):
    """Chat request model"""
    query: str = Field(..., description="User query about clinical trials", min_length=1, max_length=1000)
    n_results: Optional[int] = Field(5, description="Number of trials to retrieve", ge=1, le=20)
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "Are there any clinical trials for PTSD?",
                "n_results": 5
            }
        }


class Source(BaseModel):
    """Source information model"""
    NCTId: str
    Title: str
    Score: str
    Relevance: str


class Timing(BaseModel):
    """Timing information model"""
    retrieval_time: str
    api_time: str
    total_time: str


class ChatResponse(BaseModel):
    """Chat response model"""
    success: bool
    query: str
    answer: str
    sources: List[Source]
    timing: Timing
    timestamp: float
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "query": "Are there any clinical trials for PTSD?",
                "answer": "Yes, there are clinical trials for PTSD. Trial NCT05812131 (COPEWeb Training for Providers)...",
                "sources": [
                    {
                        "NCTId": "NCT05812131",
                        "Title": "COPEWeb Training for Providers",
                        "Score": "15.23",
                        "Relevance": "100.0%"
                    }
                ],
                "timing": {
                    "retrieval_time": "0.123s",
                    "api_time": "1.456s",
                    "total_time": "1.579s"
                },
                "timestamp": 1234567890.123
            }
        }


class StatsResponse(BaseModel):
    """Statistics response model"""
    total_queries: int
    avg_retrieval_time: str
    avg_api_time: str
    avg_total_time: str
    total_retrieval_time: str
    total_api_time: str
    total_time: str


class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    service: str
    version: str
    chatbot_loaded: bool
    timestamp: float


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    message: str
    timestamp: float


# API Endpoints
@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information"""
    return {
        "service": "Clinical Trial RAG Chatbot API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "chat": "/chat (POST)",
            "stats": "/stats",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if chatbot is not None else "unhealthy",
        service="Clinical Trial RAG Chatbot API",
        version="1.0.0",
        chatbot_loaded=chatbot is not None,
        timestamp=time.time()
    )


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Chat endpoint - Query clinical trials
    
    - query: Natural language question about clinical trials
    - n_results: Number of results to retrieve (default: 5, max: 20)
    
    Returns answer with sources and timing information
    """
    if chatbot is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chatbot not initialized"
        )
    
    try:
        logger.info(f"Processing query: {request.query}")
        
        # Process query
        result = chatbot.chat(request.query)
        
        # Convert sources to Source model
        sources = [Source(**source) for source in result['sources']]
        
        # Convert timing to Timing model
        timing = Timing(**result['timing'])
        
        response = ChatResponse(
            success=True,
            query=request.query,
            answer=result['answer'],
            sources=sources,
            timing=timing,
            timestamp=time.time()
        )
        
        logger.info(f"Query processed successfully in {result['timing']['total_time']}")
        return response
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing query: {str(e)}"
        )


@app.get("/stats", response_model=StatsResponse, tags=["Statistics"])
async def get_statistics():
    """
    Get chatbot statistics
    
    Returns statistics about queries processed including:
    - Total number of queries
    - Average timing metrics
    - Total timing metrics
    """
    if chatbot is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chatbot not initialized"
        )
    
    try:
        stats = chatbot.get_stats()
        return StatsResponse(**stats)
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting statistics: {str(e)}"
        )


@app.post("/reset-stats", tags=["Statistics"])
async def reset_statistics():
    """
    Reset chatbot statistics
    
    Resets all statistics counters to zero
    """
    if chatbot is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chatbot not initialized"
        )
    
    try:
        chatbot.stats = {
            'total_queries': 0,
            'total_retrieval_time': 0,
            'total_api_time': 0,
            'total_time': 0
        }
        return {
            "success": True,
            "message": "Statistics reset successfully",
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Error resetting statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error resetting statistics: {str(e)}"
        )


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Handle 404 errors"""
    return {
        "error": "Not Found",
        "message": f"The requested endpoint does not exist",
        "timestamp": time.time()
    }


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {exc}")
    return {
        "error": "Internal Server Error",
        "message": "An internal error occurred",
        "timestamp": time.time()
    }


if __name__ == "__main__":
    import uvicorn
    
    # Run the API server
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enable auto-reload during development
        log_level="info"
    )

