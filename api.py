from fastapi import FastAPI, HTTPException, BackgroundTasks
import uvicorn
from pydantic import BaseModel, Field, EmailStr
import logging
from typing import List, Dict, Optional, Any
import os
import re
import nltk
from nltk.tokenize import sent_tokenize
import time
import requests
import json
from concurrent.futures import ThreadPoolExecutor
import hashlib
import pickle
from fastapi.middleware.cors import CORSMiddleware
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Try to load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    logging.info("Loaded environment variables from .env file")
except ImportError:
    logging.warning("python-dotenv not installed, skipping .env file loading")

# Import the necessary functions from extract_meeting_info.py
from scripts.extract_meeting_info import (
    extract_meeting_info,
    process_with_ollama,
    parse_response
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to download NLTK data if not present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# Create FastAPI app
app = FastAPI(
    title="Meeting Transcript Analyzer API",
    description="API for extracting summaries, decisions, and action items from meeting transcripts",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache directory
CACHE_DIR = os.path.join("models", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Models
class TranscriptRequest(BaseModel):
    text: str = Field(..., description="Raw meeting transcript text")
    model: str = Field(default="mistral", description="Ollama model to use (e.g., mistral, llama3.2)")
    max_chunk_size: int = Field(default=1500, description="Maximum number of characters per chunk")
    use_cache: bool = Field(default=True, description="Whether to use caching for responses")

class MeetingItem(BaseModel):
    summary: List[str] = Field(default_factory=list, description="List of summary points")
    decisions: List[str] = Field(default_factory=list, description="List of decisions made")
    action_items: List[str] = Field(default_factory=list, description="List of action items")

class TranscriptResponse(BaseModel):
    success: bool = Field(..., description="Whether the processing was successful")
    message: str = Field(..., description="Status message")
    results: MeetingItem = Field(..., description="Extracted meeting information")
    processing_time: float = Field(..., description="Processing time in seconds")
    model_used: str = Field(..., description="Model used for processing")
    fallback_used: bool = Field(..., description="Whether fallback extraction was used")

class EmailMetadata(BaseModel):
    model: str
    processing_time: float
    fallback_used: bool

class EmailContent(BaseModel):
    summary: List[str]
    decisions: List[str]
    action_items: List[str]
    metadata: EmailMetadata

class EmailRequest(BaseModel):
    to: EmailStr
    subject: str = "Meeting Analysis Results"
    content: EmailContent

# Helper functions
def chunk_text(text: str, max_chunk_size: int = 1500) -> List[str]:
    """
    Split text into chunks of approximately max_chunk_size characters,
    trying to break at sentence boundaries.
    """
    # If text is short enough, return as is
    if len(text) <= max_chunk_size:
        return [text]
    
    # Tokenize into sentences
    try:
        sentences = sent_tokenize(text)
    except Exception:
        # Fallback to simple splitting if NLTK fails
        sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # If adding this sentence would exceed max_chunk_size,
        # start a new chunk (unless current_chunk is empty)
        if current_chunk and len(current_chunk) + len(sentence) > max_chunk_size:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk += " " + sentence if current_chunk else sentence
    
    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def get_cache_key(text: str, model: str) -> str:
    """Generate a cache key based on text content and model."""
    hash_obj = hashlib.md5(f"{text}:{model}".encode())
    return hash_obj.hexdigest()

def save_to_cache(key: str, data: Any) -> None:
    """Save data to cache."""
    cache_path = os.path.join(CACHE_DIR, f"{key}.pkl")
    with open(cache_path, 'wb') as f:
        pickle.dump(data, f)

def load_from_cache(key: str) -> Optional[Any]:
    """Load data from cache if it exists."""
    cache_path = os.path.join(CACHE_DIR, f"{key}.pkl")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading cache: {e}")
    return None

def process_transcript_chunk(chunk: str, model: str) -> Dict[str, Any]:
    """Process a single transcript chunk."""
    try:
        # Try using Ollama with longer timeout for phi model
        if model == "phi":
            logger.info(f"Using phi model with extended timeout")
        response = process_with_ollama(chunk, model)
        parsed = parse_response(response)
        fallback_used = False
    except Exception as e:
        logger.error(f"Error processing with Ollama: {e}")
        # Fallback to rule-based extraction
        parsed = extract_meeting_info(chunk)
        fallback_used = True
    
    return {
        "parsed": parsed,
        "fallback_used": fallback_used,
        "raw_response": response if 'response' in locals() else None
    }

def merge_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge results from multiple chunks."""
    merged = {
        "summary": [],
        "decisions": [],
        "action_items": [],
        "fallback_used": any(r["fallback_used"] for r in results)
    }
    
    # Merge all results
    for result in results:
        parsed = result["parsed"]
        merged["summary"].extend(parsed.get("summary", []))
        merged["decisions"].extend(parsed.get("decisions", []))
        merged["action_items"].extend(parsed.get("action_items", []))
    
    # Deduplicate
    merged["summary"] = list(dict.fromkeys(merged["summary"]))
    merged["decisions"] = list(dict.fromkeys(merged["decisions"]))
    merged["action_items"] = list(dict.fromkeys(merged["action_items"]))
    
    return merged

def send_email_notification(email_request: EmailRequest, background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """Send email with meeting analysis results."""
    background_tasks.add_task(_send_email, email_request)
    return {"success": True, "message": "Email scheduled for delivery"}

async def _send_email(email_request: EmailRequest) -> None:
    """Send email in the background."""
    try:
        # Create email message
        message = MIMEMultipart("alternative")
        message["Subject"] = email_request.subject
        message["From"] = os.environ.get("EMAIL_FROM", "meeting-analyzer@example.com")
        message["To"] = email_request.to
        
        # Create HTML content
        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                h1 {{ color: #2c3e50; border-bottom: 1px solid #eee; padding-bottom: 10px; }}
                h2 {{ color: #3498db; margin-top: 20px; }}
                ul {{ margin-bottom: 20px; }}
                li {{ margin-bottom: 8px; }}
                .metadata {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-top: 30px; }}
                .footer {{ margin-top: 30px; font-size: 12px; color: #777; text-align: center; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Meeting Analysis Results</h1>
                
                <h2>Summary</h2>
                <ul>
                    {"".join([f"<li>{item}</li>" for item in email_request.content.summary])}
                </ul>
                
                <h2>Decisions</h2>
                <ul>
                    {
                        "".join([f"<li>{item}</li>" for item in email_request.content.decisions]) 
                        if email_request.content.decisions 
                        else "<li>None identified</li>"
                    }
                </ul>
                
                <h2>Action Items</h2>
                <ul>
                    {
                        "".join([f"<li>{item}</li>" for item in email_request.content.action_items]) 
                        if email_request.content.action_items 
                        else "<li>None identified</li>"
                    }
                </ul>
                
                <div class="metadata">
                    <p><strong>Model used:</strong> {email_request.content.metadata.model}</p>
                    <p><strong>Processing time:</strong> {email_request.content.metadata.processing_time:.2f} seconds</p>
                    <p><strong>Fallback used:</strong> {"Yes" if email_request.content.metadata.fallback_used else "No"}</p>
                </div>
                
                <div class="footer">
                    <p>This is an automated email from the Meeting Transcript Analyzer.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Attach HTML content
        part = MIMEText(html, "html")
        message.attach(part)
        
        # Also create plain text version
        text_content = f"""
Meeting Analysis Results

SUMMARY:
{"".join(['- ' + item + '\n' for item in email_request.content.summary])}

DECISIONS:
{"".join(['- ' + item + '\n' for item in email_request.content.decisions]) if email_request.content.decisions else "None identified\n"}

ACTION ITEMS:
{"".join(['- ' + item + '\n' for item in email_request.content.action_items]) if email_request.content.action_items else "None identified\n"}

Model used: {email_request.content.metadata.model}
Processing time: {email_request.content.metadata.processing_time:.2f} seconds
Fallback used: {"Yes" if email_request.content.metadata.fallback_used else "No"}

This is an automated email from the Meeting Transcript Analyzer.
        """
        text_part = MIMEText(text_content, "plain")
        message.attach(text_part)
        
        # Get email configuration from environment variables
        smtp_server = os.environ.get("SMTP_SERVER")
        smtp_port = int(os.environ.get("SMTP_PORT", 587))
        smtp_username = os.environ.get("SMTP_USERNAME")
        smtp_password = os.environ.get("SMTP_PASSWORD")
        
        # Check if SMTP configuration is available
        if smtp_server and smtp_username and smtp_password:
            # Send email using SMTP
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(smtp_username, smtp_password)
                server.send_message(message)
                logger.info(f"Email sent successfully to {email_request.to}")
        else:
            # Log the email content if SMTP is not configured
            logger.info(f"SMTP not configured. Would send email to {email_request.to} with subject: {email_request.subject}")
            logger.info(f"Email content: {html[:100]}...")
            logger.warning("To send actual emails, set SMTP_SERVER, SMTP_USERNAME, and SMTP_PASSWORD environment variables")
        
    except Exception as e:
        logger.error(f"Error sending email: {e}")
        # Raise the exception to be handled by the caller
        raise

# API endpoints
@app.post("/analyze", response_model=TranscriptResponse)
async def analyze_transcript(request: TranscriptRequest):
    """
    Analyze a meeting transcript and extract summary, decisions, and action items.
    
    If the transcript is too long, it will be split into chunks and processed in parallel.
    """
    start_time = time.time()
    
    # Check if text is empty
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Transcript text cannot be empty")
    
    # Check cache if enabled
    if request.use_cache:
        cache_key = get_cache_key(request.text, request.model)
        cached_result = load_from_cache(cache_key)
        if cached_result:
            logger.info(f"Using cached result for {cache_key}")
            processing_time = time.time() - start_time
            return TranscriptResponse(
                success=True,
                message="Analysis completed (cached)",
                results=cached_result["results"],
                processing_time=processing_time,
                model_used=request.model,
                fallback_used=cached_result["fallback_used"]
            )
    
    # Chunk the text if necessary
    chunks = chunk_text(request.text, request.max_chunk_size)
    logger.info(f"Split transcript into {len(chunks)} chunks")
    
    # Process chunks
    results = []
    fallback_used = False
    
    # Process in parallel for multiple chunks
    if len(chunks) > 1:
        with ThreadPoolExecutor(max_workers=min(len(chunks), 4)) as executor:
            futures = [executor.submit(process_transcript_chunk, chunk, request.model) for chunk in chunks]
            for future in futures:
                try:
                    result = future.result()
                    results.append(result)
                    if result["fallback_used"]:
                        fallback_used = True
                except Exception as e:
                    logger.error(f"Error processing chunk: {e}")
                    # If a chunk fails, use rule-based extraction as fallback
                    fallback_used = True
    else:
        # Process single chunk directly
        result = process_transcript_chunk(chunks[0], request.model)
        results.append(result)
        fallback_used = result["fallback_used"]
    
    # Merge results from all chunks
    merged = merge_results(results)
    
    # Create response
    meeting_item = MeetingItem(
        summary=merged["summary"],
        decisions=merged["decisions"],
        action_items=merged["action_items"]
    )
    
    processing_time = time.time() - start_time
    
    # Cache the result if enabled
    if request.use_cache:
        cache_data = {
            "results": meeting_item,
            "fallback_used": merged["fallback_used"]
        }
        save_to_cache(cache_key, cache_data)
    
    return TranscriptResponse(
        success=True,
        message="Analysis completed successfully" if not fallback_used else "Analysis completed with fallback extraction",
        results=meeting_item,
        processing_time=processing_time,
        model_used=request.model,
        fallback_used=merged["fallback_used"]
    )

@app.post("/send-email")
async def send_email(request: EmailRequest, background_tasks: BackgroundTasks):
    """
    Send an email with the meeting analysis results.
    
    The email will be sent asynchronously in the background.
    """
    try:
        result = send_email_notification(request, background_tasks)
        return result
    except Exception as e:
        logger.error(f"Error sending email: {e}")
        raise HTTPException(status_code=500, detail=f"Error sending email: {str(e)}")

@app.get("/models")
async def list_models():
    """
    List available Ollama models.
    """
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = response.json().get("models", [])
            return {"models": [model["name"] for model in models]}
        else:
            # Fallback models if Ollama is not available
            return {"models": ["mistral", "llama3.2", "gemma", "phi"]}
    except Exception:
        # Fallback models if Ollama is not available
        return {"models": ["mistral", "llama3.2", "gemma", "phi"]}

@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    """
    return {"status": "ok", "timestamp": time.time()}

@app.delete("/cache")
async def clear_cache():
    """
    Clear all cached analysis results.
    """
    try:
        # Count files before deletion
        cache_files = [f for f in os.listdir(CACHE_DIR) if f.endswith('.pkl')]
        file_count = len(cache_files)
        
        # Delete all cache files
        for filename in cache_files:
            file_path = os.path.join(CACHE_DIR, filename)
            try:
                os.remove(file_path)
            except Exception as e:
                logger.error(f"Error deleting cache file {filename}: {e}")
        
        return {
            "success": True, 
            "message": f"Successfully cleared {file_count} cache files",
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(status_code=500, detail=f"Error clearing cache: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)