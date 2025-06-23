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
    llm_type: str = Field(default="ollama", description="LLM type to use (ollama, openai, lmstudio, rule-based)")
    api_url: Optional[str] = Field(default=None, description="API URL for OpenAI-compatible APIs")
    api_key: Optional[str] = Field(default=None, description="API key for OpenAI-compatible APIs")
    disable_chunking: bool = Field(default=True, description="Process the entire transcript as a single chunk (default: True)")

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
    
    # Get LLM type and model
    llm_type = request.llm_type
    model = request.model
    
    # Check cache if enabled
    if request.use_cache:
        # Include LLM type in cache key to differentiate between backends
        cache_key = get_cache_key(f"{request.text}:{llm_type}", model)
        cached_result = load_from_cache(cache_key)
        if cached_result:
            logger.info(f"Using cached result for {cache_key}")
            processing_time = time.time() - start_time
            return TranscriptResponse(
                success=True,
                message="Analysis completed (cached)",
                results=cached_result["results"],
                processing_time=processing_time,
                model_used=f"{llm_type}:{model}",
                fallback_used=cached_result["fallback_used"]
            )
    
    # Chunk the text if necessary, unless chunking is disabled
    if request.disable_chunking:
        chunks = [request.text]
        logger.info("Processing entire transcript as a single chunk (chunking disabled)")
    else:
        chunks = chunk_text(request.text, request.max_chunk_size)
        logger.info(f"Split transcript into {len(chunks)} chunks")
    
    # Process chunks
    results = []
    fallback_used = False
    
    # Process in parallel for multiple chunks
    if len(chunks) > 1:
        with ThreadPoolExecutor(max_workers=min(len(chunks), 4)) as executor:
            futures = []
            for chunk in chunks:
                if llm_type == "ollama":
                    futures.append(executor.submit(process_with_ollama, chunk, model))
                elif llm_type == "openai":
                    futures.append(executor.submit(process_with_openai_compatible, chunk, request.api_url, request.api_key, request.model))
                elif llm_type == "lmstudio":
                    futures.append(executor.submit(process_with_lmstudio, chunk, request.api_url))
                else:  # rule-based
                    futures.append(executor.submit(extract_meeting_info, chunk))
            
            for future in futures:
                try:
                    if llm_type == "rule-based":
                        # For rule-based, we get the parsed result directly
                        parsed = future.result()
                        result = {
                            "parsed": parsed,
                            "fallback_used": True,
                            "raw_response": None
                        }
                    else:
                        # For LLM-based, we get the response and parse it
                        response = future.result()
                        result = process_transcript_chunk_response(response, chunk, llm_type)
                    
                    results.append(result)
                    if result["fallback_used"]:
                        fallback_used = True
                except Exception as e:
                    logger.error(f"Error processing chunk: {e}")
                    # If a chunk fails, use rule-based extraction as fallback
                    fallback_used = True
    else:
        # Process single chunk directly
        try:
            if llm_type == "ollama":
                response = process_with_ollama(chunks[0], model)
                result = process_transcript_chunk_response(response, chunks[0], llm_type)
            elif llm_type == "openai":
                response = process_with_openai_compatible(chunks[0], request.api_url, request.api_key, request.model)
                result = process_transcript_chunk_response(response, chunks[0], llm_type)
            elif llm_type == "lmstudio":
                response = process_with_lmstudio(chunks[0], request.api_url)
                result = process_transcript_chunk_response(response, chunks[0], llm_type)
            else:  # rule-based
                parsed = extract_meeting_info(chunks[0])
                result = {
                    "parsed": parsed,
                    "fallback_used": True,
                    "raw_response": None
                }
            
            results.append(result)
            fallback_used = result["fallback_used"]
        except Exception as e:
            logger.error(f"Error processing chunk: {e}")
            # If processing fails, use rule-based extraction as fallback
            parsed = extract_meeting_info(chunks[0])
            results.append({
                "parsed": parsed,
                "fallback_used": True,
                "raw_response": None
            })
            fallback_used = True
    
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
        cache_key = get_cache_key(f"{request.text}:{llm_type}", model)
        save_to_cache(cache_key, cache_data)
    
    return TranscriptResponse(
        success=True,
        message="Analysis completed successfully" if not fallback_used else "Analysis completed with fallback extraction",
        results=meeting_item,
        processing_time=processing_time,
        model_used=f"{llm_type}:{model}",
        fallback_used=merged["fallback_used"]
    )

def process_transcript_chunk_response(response: str, chunk: str, llm_type: str) -> Dict[str, Any]:
    """Helper function to process a response from an LLM."""
    try:
        # Parse the response
        parse_response.current_transcript = chunk  # Store for fallback
        parsed = parse_response(response)
        fallback_used = False
    except Exception as e:
        logger.error(f"Error processing with {llm_type}: {e}")
        # Fallback to rule-based extraction
        parsed = extract_meeting_info(chunk)
        fallback_used = True
    
    return {
        "parsed": parsed,
        "fallback_used": fallback_used,
        "raw_response": response if 'response' in locals() else None
    }

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

def fix_malformed_json(json_str: str) -> str:
    """
    Attempt to fix common issues with malformed JSON returned by LLMs.
    """
    # Remove any text before the first opening brace
    json_str = re.sub(r'^[^{]*', '', json_str)
    
    # Remove any text after the last closing brace
    json_str = re.sub(r'[^}]*$', '', json_str)
    
    # Fix missing quotes around keys
    json_str = re.sub(r'(\s*?)(\w+)(\s*?):', r'\1"\2"\3:', json_str)
    
    # Fix trailing commas in objects
    json_str = re.sub(r',\s*}', '}', json_str)
    
    # Fix trailing commas in arrays
    json_str = re.sub(r',\s*]', ']', json_str)
    
    # Fix missing quotes around string values
    def fix_unquoted_strings(match):
        key = match.group(1)
        value = match.group(2)
        if value.lower() in ['true', 'false', 'null'] or re.match(r'^-?\d+(\.\d+)?$', value):
            # Don't quote booleans, null, or numbers
            return f'"{key}": {value}'
        else:
            # Quote other values
            return f'"{key}": "{value}"'
    
    json_str = re.sub(r'"(\w+)":\s*([^",\{\[\]\}\s][^",\{\[\]\}\s]*)', fix_unquoted_strings, json_str)
    
    # Replace nested JSON objects with string placeholders
    json_str = re.sub(r'("summary":\s*){([^}]*)}', r'\1"NESTED_JSON"', json_str)
    
    return json_str

def process_with_ollama(text: str, model: str) -> str:
    """
    Process text with Ollama LLM.
    """
    from scripts.extract_meeting_info import process_with_ollama as extract_process_with_ollama
    
    # Maximum number of retries
    max_retries = 3
    retry_delay = 2  # seconds
    
    for attempt in range(max_retries):
        try:
            # Use a much longer timeout for larger documents
            response = extract_process_with_ollama(text, model)
            
            # Try to fix malformed JSON if present
            if '{"summary":' in response and ('"decisions":' in response or '"action_items":' in response):
                try:
                    # Extract JSON-like content
                    json_pattern = re.compile(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```|(\{[\s\S]*?\})')
                    json_matches = json_pattern.findall(response)
                    
                    if json_matches:
                        for json_match in json_matches:
                            # Take the non-empty match from the tuple
                            json_str = json_match[0] if json_match[0] else json_match[1]
                            if json_str and '{"summary":' in json_str:
                                # Try to fix and parse the JSON
                                fixed_json = fix_malformed_json(json_str)
                                try:
                                    # Test if it's valid JSON now
                                    json.loads(fixed_json)
                                    # If valid, replace the original JSON in the response
                                    response = response.replace(json_str, fixed_json)
                                except json.JSONDecodeError:
                                    # If still invalid, keep the original
                                    pass
                except Exception as e:
                    logger.warning(f"Error fixing JSON: {e}")
            
            return response
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama API request error (attempt {attempt+1}/{max_retries}): {e}")
            
            # If not the last attempt, retry
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                # If all retries failed, fall back to rule-based extraction
                logger.warning("All Ollama API attempts failed, falling back to rule-based extraction")
                # Use rule-based extraction as fallback
                parsed = extract_meeting_info(text)
                
                # Format the summary as a paragraph
                summary_paragraph = " ".join(parsed["summary"])
                
                return f"""## SUMMARY
{summary_paragraph}

## DECISIONS
{chr(10).join([f"- {item}" for item in parsed["decisions"]]) if parsed["decisions"] else "- None identified"}

## ACTION ITEMS
{chr(10).join([f"- {item}" for item in parsed["action_items"]]) if parsed["action_items"] else "- None identified"}"""
        except Exception as e:
            logger.error(f"Unexpected error with Ollama (attempt {attempt+1}/{max_retries}): {e}")
            
            # If not the last attempt, retry
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                # If all retries failed, fall back to rule-based extraction
                logger.warning("All Ollama attempts failed, falling back to rule-based extraction")
                # Use rule-based extraction as fallback
                parsed = extract_meeting_info(text)
                
                # Format the summary as a paragraph
                summary_paragraph = " ".join(parsed["summary"])
                
                return f"""## SUMMARY
{summary_paragraph}

## DECISIONS
{chr(10).join([f"- {item}" for item in parsed["decisions"]]) if parsed["decisions"] else "- None identified"}

## ACTION ITEMS
{chr(10).join([f"- {item}" for item in parsed["action_items"]]) if parsed["action_items"] else "- None identified"}"""

def process_with_openai_compatible(text: str, api_url: str, api_key: str, model: str = "gpt-3.5-turbo") -> str:
    """Process a chunk of text with OpenAI API or compatible endpoints."""
    try:
        # Use default OpenAI URL if not provided
        if not api_url:
            api_url = "https://api.openai.com/v1/chat/completions"
        
        # Prepare the prompt
        system_message = """
        You are an AI assistant that extracts information from meeting transcripts.
        Analyze the meeting transcript and extract:
        1. A concise summary of the meeting (key points discussed)
        2. All decisions made during the meeting
        3. All action items/tasks assigned during the meeting, including who is responsible and any deadlines
        
        Format your response as follows:
        ```json
        {
            "summary": ["point 1", "point 2", ...],
            "decisions": ["decision 1", "decision 2", ...],
            "action_items": ["action item 1", "action item 2", ...]
        }
        ```
        
        If there are no decisions or action items, use an empty array [].
        Be concise and focus only on extracting the requested information.
        """
        
        user_message = f"TRANSCRIPT:\n{text}"
        
        # Prepare request payload
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            "temperature": 0.3,
            "max_tokens": 1000
        }
        
        # Set headers with API key
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        # Call OpenAI API
        response = requests.post(
            api_url,
            headers=headers,
            json=payload,
            timeout=270  # 270 second timeout
        )
        
        if response.status_code != 200:
            logger.error(f"OpenAI API error: {response.status_code} - {response.text}")
            raise Exception(f"OpenAI API error: {response.status_code}")
        
        result = response.json()
        # Extract content from the response
        if "choices" in result and len(result["choices"]) > 0:
            if "message" in result["choices"][0] and "content" in result["choices"][0]["message"]:
                return result["choices"][0]["message"]["content"]
        
        logger.error(f"Unexpected OpenAI API response format: {result}")
        raise Exception("Unexpected OpenAI API response format")
    except Exception as e:
        logger.error(f"Error calling OpenAI API: {e}")
        raise

def process_with_lmstudio(text: str, api_url: str) -> str:
    """Process a chunk of text with LM Studio API."""
    try:
        # Use default URL if not provided
        if not api_url:
            api_url = "http://localhost:1234/v1/chat/completions"
        
        # Prepare the prompt
        system_message = """
        You are an AI assistant that extracts information from meeting transcripts.
        Analyze the meeting transcript and extract:
        1. A concise summary of the meeting (key points discussed)
        2. All decisions made during the meeting
        3. All action items/tasks assigned during the meeting, including who is responsible and any deadlines
        
        Format your response as follows:
        ```json
        {
            "summary": ["point 1", "point 2", ...],
            "decisions": ["decision 1", "decision 2", ...],
            "action_items": ["action item 1", "action item 2", ...]
        }
        ```
        
        If there are no decisions or action items, use an empty array [].
        Be concise and focus only on extracting the requested information.
        """
        
        user_message = f"TRANSCRIPT:\n{text}"
        
        # Prepare request payload
        payload = {
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            "temperature": 0.3,
            "max_tokens": 1000
        }
        
        # Call LM Studio API
        response = requests.post(
            api_url,
            json=payload,
            timeout=270  # 270 second timeout
        )
        
        if response.status_code != 200:
            logger.error(f"LM Studio API error: {response.status_code} - {response.text}")
            raise Exception(f"LM Studio API error: {response.status_code}")
        
        result = response.json()
        # Extract content from the response (same format as OpenAI)
        if "choices" in result and len(result["choices"]) > 0:
            if "message" in result["choices"][0] and "content" in result["choices"][0]["message"]:
                return result["choices"][0]["message"]["content"]
        
        logger.error(f"Unexpected LM Studio API response format: {result}")
        raise Exception("Unexpected LM Studio API response format")
    except Exception as e:
        logger.error(f"Error calling LM Studio API: {e}")
        raise

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)