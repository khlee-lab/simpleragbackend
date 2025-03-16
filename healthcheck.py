from fastapi import FastAPI, HTTPException
import os

def verify_app_health(app: FastAPI):
    # Add a health check endpoint
    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "port": os.environ.get("PORT", "8000")}
    
    return app
