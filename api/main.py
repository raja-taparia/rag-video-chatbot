from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import os
import logging

# Lazy imports for pipeline (deferred until app startup)
from src.config import load_config
from src.pipeline.rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)

def mount_static_files():
    """Create a FastAPI app, ensure a `static` directory exists, and mount it.

    Returns:
        FastAPI: configured FastAPI application with `/static` mounted
    """
    app = FastAPI()

    # Mount a `static` directory located at the repository root (or create it)
    static_dir = Path(__file__).resolve().parent.parent / "static"
    os.makedirs(static_dir, exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    return app


# Create the app using the helper to keep this file modular
app = mount_static_files()


# Initialize RAG pipeline once on module import so requests can use it.
try:
    _config = load_config()
    _pipeline = RAGPipeline(_config)
    logger.info("RAG pipeline initialized for API")
except Exception as e:
    # If pipeline fails to initialize, keep going — endpoint will return 500.
    logger.error(f"Failed to initialize RAG pipeline at startup: {e}")
    _pipeline = None

@app.get("/", response_class=HTMLResponse)
async def web_interface():
    """Serve beautiful web interface"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>RAG Chatbot</title>
        <style>
            body { font-family: Arial; max-width: 1200px; margin: 50px auto; }
            input, textarea { width: 100%; padding: 10px; margin: 5px 0; }
            button { background: #007bff; color: white; padding: 10px 20px; cursor: pointer; }
            .response { background: #f0f0f0; padding: 15px; margin: 10px 0; border-left: 4px solid #007bff; }
            .video { background: #e8f4f8; border-left-color: #17a2b8; }
            .pdf { background: #f8f4e8; border-left-color: #ffc107; }
            .timestamp { color: #007bff; font-weight: bold; }
        </style>
    </head>
    <body>
        <h1>💬 RAG Chatbot - Ask Me Anything</h1>
        
        <textarea id="question" placeholder="Ask your question..." rows="4"></textarea>
        <button onclick="askQuestion()">🔍 Search</button>
        
        <div id="result"></div>
        
        <script>
            async function askQuestion() {
                const question = document.getElementById('question').value;
                if (!question) return;
                
                try {
                    const response = await fetch('/ask', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({question})
                    });
                    
                    const data = await response.json();
                    const resultDiv = document.getElementById('result');
                    
                    if (data.response.source_type === 'video') {
                        const start = data.response.segments[0].start_timestamp;
                        const end = data.response.segments[0].end_timestamp;
                        resultDiv.innerHTML = `
                            <div class="response video">
                                <h3>📹 Video Answer</h3>
                                <p>${data.response.answer_text}</p>
                                <p><span class="timestamp">⏱️ Timestamp: ${start}s - ${end}s</span></p>
                                <p>Video: ${data.response.title}</p>
                            </div>
                        `;
                    } else if (data.response.source_type === 'pdf') {
                        resultDiv.innerHTML = `
                            <div class="response pdf">
                                <h3>📄 PDF Answer</h3>
                                <p>${data.response.answer_text}</p>
                                <p>Source: ${data.response.pdf_filename}, Page ${data.response.page_number}</p>
                            </div>
                        `;
                    } else {
                        resultDiv.innerHTML = `
                            <div class="response">
                                <p>❌ ${data.response.message}</p>
                            </div>
                        `;
                    }
                } catch (error) {
                    console.error('Error:', error);
                }
            }
        </script>
    </body>
    </html>
    """

@app.post("/upload-video")
async def upload_video(file: UploadFile = File(...)):
    """Upload local video file"""
    try:
        contents = await file.read()
        video_path = Path(f"data/videos/videos_input/{file.filename}")
        with open(video_path, "wb") as f:
            f.write(contents)
        
        return {"status": "uploaded", "filename": file.filename}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))



@app.post("/ask")
async def ask_question(payload: dict):
    """Handle question from frontend and return RAGResponse as JSON.

    Expected payload: {"question": "..."}
    """
    try:
        question = payload.get("question") if isinstance(payload, dict) else None
        if not question:
            raise HTTPException(status_code=400, detail="Missing 'question' in request body")

        if _pipeline is None:
            raise HTTPException(status_code=500, detail="RAG pipeline not initialized")

        # Run the pipeline synchronously (it's CPU-bound and quick for small queries)
        response = _pipeline.query(question)
        return response.model_dump()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error while handling /ask: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status/videos")
async def video_status():
    """Get video transcription status"""
    videos_dir = Path("data/videos")
    videos = list(videos_dir.glob("*.json"))
    return {
        "total_videos": len(videos),
        "videos": [v.name for v in videos]
    }

@app.get("/status/pdfs")
async def pdf_status():
    """Get PDF indexing status"""
    pdfs_dir = Path("data/pdfs")
    pdfs = list(pdfs_dir.glob("*.pdf"))
    return {
        "total_pdfs": len(pdfs),
        "pdfs": [p.name for p in pdfs]
    }