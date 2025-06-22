# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import argparse
import os
import logging
from tqdm import tqdm
import json
from typing import List, Dict, Any, Optional, Iterator
import requests
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import pickle
from functools import lru_cache
import threading
import warnings
from transformers import logging as transformers_logging
import tempfile
import shutil
import nltk
from nltk.tokenize import sent_tokenize
import spacy
import string
import subprocess

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress transformer warnings
transformers_logging.set_verbosity_error()
warnings.filterwarnings('ignore')

# Global variables for NLP models
_SPACY_MODEL = None
_SPACY_LOCK = threading.Lock()

# Try to download NLTK data if not present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

def get_spacy_model():
    """Get or initialize spaCy model with thread safety."""
    global _SPACY_MODEL
    
    with _SPACY_LOCK:
        if _SPACY_MODEL is None:
            try:
                import spacy
                try:
                    _SPACY_MODEL = spacy.load("en_core_web_sm")
                    logger.info("Loaded spaCy model: en_core_web_sm")
                except OSError:
                    logger.info("Downloading spaCy model...")
                    os.system("python -m spacy download en_core_web_sm")
                    _SPACY_MODEL = spacy.load("en_core_web_sm")
            except ImportError:
                logger.warning("spaCy not available, using basic processing")
                _SPACY_MODEL = None
    
    return _SPACY_MODEL

def extract_meeting_info(transcript: str) -> Dict[str, List[str]]:
    """
    Extract meeting information using enhanced rule-based approach.
    """
    result = {
        "summary": [],
        "decisions": [],
        "action_items": []
    }
    
    # Clean transcript
    transcript = transcript.strip()
    if not transcript:
        return result
    
    # Try to use spaCy for better NLP if available
    nlp = get_spacy_model()
    if nlp:
        # Process with spaCy
        doc = nlp(transcript[:1000000])  # Limit size to avoid memory issues
        
        # Extract named entities for better context
        entities = {ent.text: ent.label_ for ent in doc.ents}
        
        # Get sentences with spaCy
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        
        # Find paragraphs (for better context)
        paragraphs = re.split(r'\n\s*\n', transcript)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
    else:
        # Fallback to regex for sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', transcript)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Simple paragraph splitting
        paragraphs = re.split(r'\n\s*\n', transcript)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        entities = {}
    
    # SUMMARY EXTRACTION
    # ------------------
    # Strategy 1: Look for explicit summary markers
    summary_markers = [
        r'summary:',
        r'meeting summary',
        r'executive summary',
        r'overview:',
        r'agenda:',
        r'purpose:',
        r'objective:',
        r'key points:',
        r'in summary',
        r'to summarize'
    ]
    
    summary_sentences = []
    
    # Look for sentences with summary markers
    for sent in sentences:
        sent_lower = sent.lower()
        if any(re.search(marker, sent_lower) for marker in summary_markers):
            # If we find a summary marker, include this sentence and the next one
            idx = sentences.index(sent)
            summary_sentences.append(sent)
            if idx + 1 < len(sentences):
                summary_sentences.append(sentences[idx + 1])
    
    # Strategy 2: Use first paragraph if it's not too long and seems like an intro
    if not summary_sentences and paragraphs:
        first_para = paragraphs[0]
        first_para_sents = re.split(r'(?<=[.!?])\s+', first_para)
        
        # Check if first paragraph is reasonably sized and doesn't look like dialogue
        if len(first_para_sents) <= 3 and len(first_para) < 500:
            # Check if it doesn't start with a speaker designation
            if not re.match(r'^[A-Z][a-z]+:', first_para):
                summary_sentences.extend(first_para_sents)
    
    # Strategy 3: Extract key sentences from the beginning and end
    if not summary_sentences:
        # Get first 2 sentences if they're substantial
        for sent in sentences[:2]:
            if len(sent) > 20 and not any(w in sent.lower() for w in ['thank', 'appreciate', 'hello', 'hi ']):
                summary_sentences.append(sent)
        
        # Get last 1-2 sentences if they seem like conclusions
        conclusion_markers = ['conclude', 'summary', 'finally', 'in conclusion', 'to sum up', 'overall']
        for sent in sentences[-3:]:
            sent_lower = sent.lower()
            if any(marker in sent_lower for marker in conclusion_markers) and len(sent) > 15:
                summary_sentences.append(sent)
    
    # Add unique summary sentences to result
    for sent in summary_sentences:
        if sent not in result["summary"] and len(sent) > 15:
            result["summary"].append(sent)
    
    # DECISION EXTRACTION
    # ------------------
    decision_indicators = [
        r'\bdecided\b', r'\bagreed\b', r'\bapproved\b', r'\badopted\b', r'\bresolution\b', 
        r'\bconcluded\b', r'\bvoted\b', r'\bconsensus\b', r'\bmotion\b', r'\bpassed\b', 
        r'\bratified\b', r'\bconfirmed\b', r'\bfinalized\b', r'\bapproved by\b', 
        r'\bsigned off\b', r'\bmoved forward with\b', r'\bdecision\b', r'\bvote\b',
        r'\bunanimously\b', r'\bmajority\b', r'\bboard approved\b', r'\bcommittee approved\b',
        r'\bthe motion carries\b', r'\bthe resolution is adopted\b'
    ]
    
    # Look for sentences with decision indicators
    for sent in sentences:
        sent_lower = sent.lower()
        
        # Check for decision indicators with more precise regex
        if any(re.search(indicator, sent_lower) for indicator in decision_indicators):
            if len(sent) > 10 and sent not in result["decisions"]:
                result["decisions"].append(sent)
    
    # Look for voting results
    vote_pattern = re.compile(r'vote(?:d|s)?\s+(?:was|is|were|are)?\s*[:;]?\s*(\d+)\s*(?:to|vs|versus|:)\s*(\d+)', re.IGNORECASE)
    for sent in sentences:
        if vote_pattern.search(sent) and sent not in result["decisions"]:
            result["decisions"].append(sent)
    
    # Look for resolution numbers
    resolution_pattern = re.compile(r'(?:resolution|motion)\s+(?:number|#|no\.?)?\s*\d+', re.IGNORECASE)
    for sent in sentences:
        if resolution_pattern.search(sent) and sent not in result["decisions"]:
            result["decisions"].append(sent)
    
    # ACTION ITEMS EXTRACTION
    # ----------------------
    # Enhanced patterns for action items
    action_patterns = [
        # Future tense patterns
        r'\bwill\s+\w+\b', r'\bshall\s+\w+\b', r'\bgoing\s+to\s+\w+\b',
        
        # Task assignment patterns
        r'\b(?:need|needs|needed)\s+to\s+\w+\b', r'\bmust\s+\w+\b', r'\bshould\s+\w+\b', 
        r'\bhave\s+to\s+\w+\b', r'\brequired\s+to\s+\w+\b', r'\bresponsible\s+for\b',
        r'\bassigned\s+to\b', r'\btask(?:ed)?\s+with\b',
        
        # Deadline patterns
        r'\bby\s+(?:the\s+)?(?:end\s+of\s+)?(?:next|this)?\s*\w+day\b',
        r'\bby\s+(?:the\s+)?(?:end\s+of\s+)?(?:january|february|march|april|may|june|july|august|september|october|november|december)\b',
        r'\bby\s+(?:the\s+)?(?:end\s+of\s+)?q[1-4]\b',
        r'\bdeadline\b', r'\bdue\s+(?:date|by)\b',
        
        # Follow-up patterns
        r'\bfollow(?:\s*-\s*|\s+)up\b', r'\bcheck\s+(?:in|back)\b', r'\breview\b',
        r'\bupdate\b', r'\bprovide\b', r'\bprepare\b', r'\bcreate\b', r'\bdevelop\b',
        
        # Specific action verbs in future context
        r'\bschedule\b', r'\barrange\b', r'\borganize\b', r'\bcoordinate\b',
        r'\bimplement\b', r'\bexecute\b', r'\bcomplete\b', r'\bfinalize\b',
        r'\bsubmit\b', r'\bdeliver\b', r'\bpresent\b', r'\bshare\b'
    ]
    
    # Look for action items with enhanced patterns
    for sent in sentences:
        sent_lower = sent.lower()
        
        # Check for action patterns
        if any(re.search(pattern, sent_lower) for pattern in action_patterns) and len(sent) > 15:
            # Check if the sentence contains a person name or role (better action item)
            has_person = False
            if nlp:
                # Use spaCy NER
                sent_doc = nlp(sent)
                for ent in sent_doc.ents:
                    if ent.label_ in ['PERSON', 'ORG']:
                        has_person = True
                        break
            
            # Higher priority for sentences with person names
            if has_person or any(re.search(r'\b(?:team|group|department|committee|everyone|all|we|you|they)\b', sent_lower)):
                if sent not in result["action_items"]:
                    result["action_items"].append(sent)
            elif sent not in result["action_items"]:
                result["action_items"].append(sent)
    
    # Look for explicit action item markers
    action_markers = [
        r'action\s+item', r'next\s+step', r'to-do', r'todo', r'task', 
        r'action\s+point', r'takeaway', r'follow\s*-?\s*up'
    ]
    
    for sent in sentences:
        sent_lower = sent.lower()
        if any(re.search(marker, sent_lower) for marker in action_markers) and len(sent) > 10:
            if sent not in result["action_items"]:
                result["action_items"].append(sent)
    
    # Look for scheduling information with enhanced pattern
    schedule_pattern = re.compile(
        r'(?:scheduled|planning|planned|set\s+up|arranged|meet|meeting|session|call)'
        r'.*?'
        r'(?:on|for|at)'
        r'\s+'
        r'(?:'
        r'(?:[A-Z][a-z]+day\s+)?'  # Day of week
        r'(?:[A-Z][a-z]+\s+\d{1,2}(?:st|nd|rd|th)?)'  # Month and day
        r'|'
        r'(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4})'  # Date in numeric format
        r'|'
        r'(?:next\s+(?:week|month|quarter))'  # Relative time
        r')'
        r'(?:\s+at\s+\d{1,2}(?::\d{2})?\s*(?:am|pm|AM|PM))?',  # Optional time
        re.IGNORECASE
    )
    
    for sent in sentences:
        schedule_match = schedule_pattern.search(sent)
        if schedule_match and sent not in result["action_items"]:
            result["action_items"].append(sent)
    
    # If we couldn't extract a summary, use a smarter fallback
    if not result["summary"]:
        # Try to find the most important sentence using basic TF-IDF principles
        if sentences:
            # Get all words
            all_words = ' '.join(sentences).lower()
            # Remove stopwords
            stopwords = {'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what', 
                        'when', 'where', 'how', 'who', 'which', 'this', 'that', 'these', 'those',
                        'then', 'just', 'so', 'than', 'such', 'both', 'through', 'about', 'for',
                        'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
                        'do', 'does', 'did', 'to', 'from', 'in', 'out', 'on', 'off', 'over', 'under',
                        'again', 'further', 'then', 'once', 'here', 'there', 'when', 'why', 'how',
                        'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
                        'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'can',
                        'will', 'should', 'now', 'I', 'you', 'he', 'she', 'we', 'they', 'it', 'its'}
            
            # Count word frequencies
            word_counts = {}
            for word in re.findall(r'\b\w+\b', all_words):
                if word not in stopwords and len(word) > 2:
                    word_counts[word] = word_counts.get(word, 0) + 1
            
            # Score sentences based on important word frequency
            sentence_scores = []
            for sent in sentences:
                score = 0
                for word in re.findall(r'\b\w+\b', sent.lower()):
                    if word in word_counts:
                        score += word_counts[word]
                sentence_scores.append((score, sent))
            
            # Get top 1-2 sentences
            sentence_scores.sort(reverse=True)
            for _, sent in sentence_scores[:2]:
                if len(sent) > 20 and sent not in result["summary"]:
                    result["summary"].append(sent)
        
        # If still empty, use a generic summary
        if not result["summary"]:
            words = transcript.split()
            if words:
                topic = ' '.join(words[:min(10, len(words))]) + "..."
                result["summary"].append(f"The transcript discusses {topic}")
    
    return result

def process_single_transcript(args):
    """
    Process a single transcript.
    """
    transcript, row_id, llm_type, model_name, api_url, api_key = args
    
    # Process transcript based on LLM type
    if llm_type == "ollama":
        response = process_with_ollama(transcript, model_name)
    elif llm_type == "lmstudio":
        response = process_with_lmstudio(transcript, api_url)
    elif llm_type == "openai":
        response = process_with_openai_compatible(transcript, api_url, api_key)
    elif llm_type == "rule-based":
        # Use rule-based extraction as fallback
        parsed = extract_meeting_info(transcript)
        raw_response = f"""## SUMMARY
{chr(10).join([f"- {item}" for item in parsed["summary"]])}

## DECISIONS
{chr(10).join([f"- {item}" for item in parsed["decisions"]]) if parsed["decisions"] else "None identified"}

## ACTION ITEMS
{chr(10).join([f"- {item}" for item in parsed["action_items"]]) if parsed["action_items"] else "None identified"}"""
        return row_id, transcript, parsed, raw_response
    else:
        response = f"Error: Unknown LLM type {llm_type}"
    
    # Store the current transcript in the parse_response function for fallback
    parse_response.current_transcript = transcript
    
    # Parse response
    parsed = parse_response(response)
    
    return row_id, transcript, parsed, response

def chunk_dataframe(df: pd.DataFrame, chunk_size: int) -> Iterator[pd.DataFrame]:
    """
    Generator to yield chunks of the dataframe.
    """
    for start_idx in range(0, len(df), chunk_size):
        yield df.iloc[start_idx:start_idx + chunk_size]

def process_chunk_with_progress(chunk_df: pd.DataFrame,
                              transcript_col: str,
                              llm_type: str,
                              model_name: str,
                              api_url: Optional[str],
                              api_key: Optional[str],
                              max_workers: int,
                              pbar: tqdm) -> pd.DataFrame:
    """
    Process a chunk of transcripts with progress tracking.
    """
    # Filter valid transcripts
    valid_rows = [(row[transcript_col], row.name) for _, row in chunk_df.iterrows() 
                 if isinstance(row[transcript_col], str) and row[transcript_col].strip()]
    
    # If no valid transcripts, return empty DataFrame
    if not valid_rows:
        return pd.DataFrame()
    
    # Create processing arguments
    process_args = [
        (transcript, row_id, llm_type, model_name, api_url, api_key)
        for transcript, row_id in valid_rows
    ]
    
    chunk_results = []
    
    # Process in parallel with ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for args in process_args:
            futures.append(executor.submit(process_single_transcript, args))
        
        # Process results as they complete
        for future in as_completed(futures):
            try:
                row_id, transcript, parsed, response = future.result()
                result_row = {
                    'transcript_id': row_id,
                    'original_text': transcript,
                    'summary': parsed['summary'],
                    'decisions': parsed['decisions'],
                    'action_items': parsed['action_items'],
                    'raw_response': response
                }
                chunk_results.append(result_row)
                pbar.update(1)
            except Exception as e:
                logger.error(f"Error processing transcript: {e}")
                pbar.update(1)
    
    return pd.DataFrame(chunk_results)

def process_transcripts(df: pd.DataFrame, 
                       transcript_col: str = 'chunk_text',
                       llm_type: str = "ollama",
                       model_name: str = "mistral",
                       api_url: Optional[str] = None,
                       api_key: Optional[str] = None,
                       max_workers: int = 2,
                       chunk_size: int = 10,
                       output_path: str = 'results/meeting_extractions.csv') -> pd.DataFrame:
    """
    Process transcripts in chunks with LLM-based extraction.
    Results are written to disk after each chunk to minimize memory usage.
    """
    # Calculate total valid transcripts for progress bar
    total_valid = sum(1 for _, row in df.iterrows() 
                     if isinstance(row[transcript_col], str) and row[transcript_col].strip())
    
    logger.info(f"Total transcripts to process: {len(df)}")
    logger.info(f"Valid transcripts to process: {total_valid}")
    logger.info(f"Processing in chunks of {chunk_size} with {max_workers} workers")
    logger.info(f"Using LLM type: {llm_type}, Model: {model_name}")
    
    # Create temporary directory for chunk results
    temp_dir = tempfile.mkdtemp(prefix="transcript_processing_")
    chunk_files = []
    chunk_count = 0
    total_processed = 0
    
    try:
        # Process in chunks with a single progress bar
        with tqdm(total=total_valid, desc="Processing transcripts") as pbar:
            for chunk_df in chunk_dataframe(df, chunk_size):
                chunk_count += 1
                logger.info(f"Processing chunk {chunk_count} with {len(chunk_df)} transcripts")
                
                # Process chunk
                chunk_results_df = process_chunk_with_progress(
                    chunk_df,
                    transcript_col,
                    llm_type,
                    model_name,
                    api_url,
                    api_key,
                    max_workers,
                    pbar
                )
                
                # Save chunk results to temporary file if we have results
                if not chunk_results_df.empty:
                    chunk_file = os.path.join(temp_dir, f"chunk_{chunk_count}.csv")
                    chunk_results_df.to_csv(chunk_file, index=False)
                    chunk_files.append(chunk_file)
                
                # Update statistics
                total_processed += len(chunk_results_df)
                logger.info(f"Completed chunk {chunk_count}. Processed {len(chunk_results_df)} transcripts")
                
                # Clear memory
                del chunk_results_df
        
        # Log processing statistics
        logger.info("\nProcessing Statistics:")
        logger.info(f"Total input transcripts: {len(df)}")
        logger.info(f"Total processed transcripts: {total_processed}")
        logger.info(f"Total chunks processed: {chunk_count}")
        
        if total_processed < total_valid:
            logger.warning(f"Warning: {total_valid - total_processed} valid transcripts were not processed!")
        
        # Combine results efficiently with low memory usage
        logger.info("Combining chunk results...")
        
        # Check if any results were processed
        if not chunk_files:
            logger.error("No results were processed!")
            # Create an empty output file with headers
            empty_df = pd.DataFrame(columns=[
                'transcript_id', 'original_text', 'summary', 'decisions', 
                'action_items', 'raw_response', 'summary_text', 
                'decisions_text', 'action_items_text'
            ])
            
            # Ensure output directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                
            # Write empty dataframe to output
            empty_df.to_csv(output_path, index=False)
            logger.info(f"Created empty output file at {output_path}")
            return pd.DataFrame()
        
        # Initialize empty DataFrame for column names
        results_df = pd.read_csv(chunk_files[0], nrows=0)
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Write header to final output file
        results_df.to_csv(output_path, index=False)
        
        # Append each chunk to the output file
        for chunk_file in tqdm(chunk_files, desc="Combining results"):
            chunk_df = pd.read_csv(chunk_file)
            
            # Convert lists to strings for CSV storage
            for col in ['summary', 'decisions', 'action_items']:
                chunk_df[f'{col}_text'] = chunk_df[col].apply(lambda x: '\n'.join([f"- {item}" for item in (eval(x) if isinstance(x, str) else x)]) if x and x != '[]' else "None identified")
            
            # Append to final file
            chunk_df.to_csv(output_path, mode='a', header=False, index=False)
            
            # Clear memory
            del chunk_df
        
        logger.info(f"Results saved to {output_path}")
        return pd.DataFrame()  # Return empty DataFrame since results are on disk
        
    finally:
        # Clean up temporary files
        logger.info("Cleaning up temporary files...")
        shutil.rmtree(temp_dir)

def check_ollama_available():
    """
    Check if Ollama is available and return available models.
    """
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        if result.returncode == 0:
            models = result.stdout.strip().split('\n')
            if models and len(models) > 1:  # Header row + at least one model
                logger.info("Ollama is available. Available models:")
                for model in models:
                    logger.info(f"  {model}")
                return True
            else:
                logger.warning("Ollama is installed but no models are available.")
                logger.info("Please pull a model with: ollama pull mistral")
                return False
        else:
            logger.warning("Ollama is installed but returned an error.")
            logger.info(result.stderr)
            return False
    except FileNotFoundError:
        logger.error("Ollama is not installed or not in PATH.")
        logger.info("Please install Ollama from https://ollama.com/download")
        logger.info("After installation, pull a model with: ollama pull mistral")
        return False

def process_with_ollama(transcript: str, model: str = "mistral") -> str:
    """
    Process transcript with Ollama LLM.
    """
    # Use a simpler prompt for smaller models
    if model in ["tinyllama", "phi", "gemma:2b"]:
        prompt = f"""You are analyzing a meeting transcript to extract key information. Read the entire transcript carefully and provide:

1. SUMMARY: Create a coherent 3-5 sentence summary of the ENTIRE transcript that captures the main topics discussed. This should be a unified summary, not disconnected points.

2. DECISIONS: List ONLY formal decisions that were actually made during the meeting, including:
   - Votes taken and their results
   - Motions passed or rejected
   - Formal agreements reached by the group
   - Policies officially adopted

3. ACTION ITEMS: List specific tasks assigned during the meeting as a TO-DO list, including:
   - WHO is responsible (always include a specific name or role)
   - WHAT specific task they need to do
   - WHEN it needs to be completed (if mentioned)
   
   Also include:
   - Scheduled meetings with specific dates/times (if no specific date/time is mentioned, don't include it)
   - Documents requiring approval with specific deadlines (if no specific deadline is mentioned, don't include it)

Transcript: {transcript}

Format your response using EXACTLY this structure with clear section separation:

## SUMMARY
A coherent paragraph summarizing the entire transcript in 3-5 sentences.

## DECISIONS
- [Specific decision that was formally made]
- [Additional decisions as needed]
(If no clear decisions were made, write only "- None identified")

## ACTION ITEMS
- [Person Name/Role]: [Specific task] by [deadline if mentioned]
- Scheduled: [Meeting details] on [specific date] at [specific time]
- Approval needed: [Document/item] by [specific deadline]
(If no action items were identified, write only "- None identified")
(Do NOT include action items without a specific person/role responsible)
"""
    else:
        prompt = f"""You are an expert meeting analyst who extracts key information from meeting transcripts with precision and clarity. Analyze the following meeting transcript and extract exactly these three elements:

1. SUMMARY:
   * Create a coherent paragraph (3-5 sentences) summarizing the entire transcript
   * Focus on substantive content, not procedural remarks
   * Capture the essence of what was discussed in a unified way
   * Ensure the summary flows logically and covers the main topics

2. DECISIONS (all that apply):
   * Extract any clear conclusions reached
   * Include formal votes, agreements, or resolutions
   * Note any approved actions, policies, or plans
   * List each decision as a separate bullet point

3. ACTION ITEMS (all that apply):
   * Create a TO-DO list format with clear ownership
   * Include WHO is responsible, WHAT they need to do, and WHEN it's due
   * Look for commitments, assignments, or follow-up tasks
   * Include any scheduled meetings with dates/times
   * Include any documents requiring approval with deadlines

For each section, if no relevant information exists, write only "None identified".

Transcript:
```
{transcript}
```

Format your response using EXACTLY this structure:

## SUMMARY
A coherent paragraph summarizing the entire transcript in 3-5 sentences, covering the main topics discussed.

## DECISIONS
- [First decision made during the meeting]
- [Additional decisions as needed]
(If no clear decisions were made, write only "- None identified")

## ACTION ITEMS
- [Person name]: [Specific task] by [Deadline if mentioned]
- Scheduled: [Meeting details] on [Date] at [Time]
- Approval needed: [Item requiring approval] by [Deadline]
(If no action items were identified, write only "- None identified")
"""

    # Maximum number of retries
    max_retries = 3
    retry_delay = 2  # seconds
    
    for attempt in range(max_retries):
        try:
            # Add increased timeout and adjust payload format
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 1000
                    }
                },
                timeout=180  # Increased timeout to 3 minutes
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "Error: No response content")
            else:
                logger.error(f"Ollama API error: {response.status_code}")
                logger.error(f"Response content: {response.text[:200]}...")
                
                # If not the last attempt, retry
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds... (Attempt {attempt+1}/{max_retries})")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                else:
                    # If all retries failed, fall back to rule-based extraction
                    logger.warning("All API attempts failed, falling back to rule-based extraction")
                    rule_based_parsed = extract_meeting_info(transcript)
                    return f"""## SUMMARY
{chr(10).join([f"- {item}" for item in rule_based_parsed["summary"]])}

## DECISIONS
{chr(10).join([f"- {item}" for item in rule_based_parsed["decisions"]]) if rule_based_parsed["decisions"] else "None identified"}

## ACTION ITEMS
{chr(10).join([f"- {item}" for item in rule_based_parsed["action_items"]]) if rule_based_parsed["action_items"] else "None identified"}"""
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {e}")
            
            # If not the last attempt, retry
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds... (Attempt {attempt+1}/{max_retries})")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                # If all retries failed, fall back to rule-based extraction
                logger.warning("All API attempts failed, falling back to rule-based extraction")
                rule_based_parsed = extract_meeting_info(transcript)
                return f"""## SUMMARY
{chr(10).join([f"- {item}" for item in rule_based_parsed["summary"]])}

## DECISIONS
{chr(10).join([f"- {item}" for item in rule_based_parsed["decisions"]]) if rule_based_parsed["decisions"] else "None identified"}

## ACTION ITEMS
{chr(10).join([f"- {item}" for item in rule_based_parsed["action_items"]]) if rule_based_parsed["action_items"] else "None identified"}"""

def process_with_openai_compatible(transcript: str, api_url: str, api_key: Optional[str] = None) -> str:
    """
    Process transcript with OpenAI-compatible API.
    """
    prompt = f"""You are an expert meeting analyst who extracts key information from meeting transcripts with precision and clarity. Analyze the following meeting transcript and extract exactly these three elements:

1. SUMMARY (3-5 bullet points):
   * Identify the main topics discussed
   * Focus on substantive content, not procedural remarks
   * Capture the essence of what was discussed, not just who spoke

2. DECISIONS (all that apply):
   * Extract any clear conclusions reached
   * Include formal votes, agreements, or resolutions
   * Note any approved actions, policies, or plans

3. ACTION ITEMS (all that apply):
   * Identify specific tasks assigned to people
   * Include WHO is responsible, WHAT they need to do, and WHEN it's due
   * Look for commitments, assignments, or follow-up tasks

For each section, if no relevant information exists, write "None identified".

Transcript:
```
{transcript}
```

Format your response using EXACTLY this structure:

## SUMMARY
- [First substantive point from the discussion]
- [Second substantive point from the discussion]
- [Additional substantive points as needed]

## DECISIONS
- [First decision made during the meeting]
- [Additional decisions as needed]

## ACTION ITEMS
- [Person name]: [Specific task] by [Deadline if mentioned]
- [Additional action items as needed]
"""

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    try:
        response = requests.post(
            api_url,
            headers=headers,
            json={
                "model": "gpt-3.5-turbo",  # This might need to be adjusted based on the API
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3
            },
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get("choices", [{}])[0].get("message", {}).get("content", "Error: No response content")
        else:
            logger.error(f"API error: {response.status_code}")
            return f"Error: API returned status code {response.status_code}"
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error: {e}")
        return f"Error: {str(e)}"

def parse_response(response: str) -> Dict[str, List[str]]:
    """
    Parse the generated response into structured data.
    
    Args:
        response: Generated response from model
        
    Returns:
        Dictionary with parsed sections
    """
    result = {
        "summary": [],
        "decisions": [],
        "action_items": []
    }
    
    # Clean up common placeholder text and prefixes
    def clean_text(text):
        if not text:
            return text
        # Remove placeholder markers
        text = re.sub(r'\[.*?\]', '', text)
        # Remove "Transcript:" prefix
        text = re.sub(r'^Transcript:', '', text)
        # Clean up extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    # First try to extract using the expected format
    # Extract summary section
    summary_match = re.search(r"##\s*SUMMARY\s*([\s\S]*?)(?=##|$)", response)
    if summary_match:
        summary_text = summary_match.group(1).strip()
        
        # First check if there are bullet points (old format)
        summary_points = re.findall(r"-\s*(.*?)(?=\n-|\n\n|$)", summary_text)
        
        if summary_points:
            # Process each summary point (bullet format)
            for point in summary_points:
                if not point.strip():
                    continue
                    
                point = point.strip()
                
                # Special handling for phi model output format
                # Check if the point contains "SUMMARY:" or "DECISIONS:" prefixes
                if point.startswith("SUMMARY:"):
                    cleaned = clean_text(point.replace("SUMMARY:", "", 1))
                    if cleaned and len(cleaned) > 5 and not all(c in '[]' for c in cleaned) and "bullet point" not in cleaned.lower():
                        result["summary"].append(cleaned)
                elif point.startswith("DECISIONS:"):
                    cleaned = clean_text(point.replace("DECISIONS:", "", 1))
                    if cleaned and len(cleaned) > 5 and not all(c in '[]' for c in cleaned) and "decision" not in cleaned.lower():
                        result["decisions"].append(cleaned)
                else:
                    cleaned = clean_text(point)
                    if cleaned and len(cleaned) > 5 and not all(c in '[]' for c in cleaned) and "bullet point" not in cleaned.lower():
                        result["summary"].append(cleaned)
        else:
            # Process as paragraph format (new format)
            # Split into sentences
            paragraph = summary_text.strip()
            if paragraph and not paragraph.lower().startswith("none identified"):
                # Remove placeholders
                paragraph = re.sub(r'\[.*?\]', '', paragraph)
                
                # Clean the paragraph
                cleaned_paragraph = clean_text(paragraph)
                
                if cleaned_paragraph and len(cleaned_paragraph) > 15:
                    # Split into sentences for better display
                    sentences = re.split(r'(?<=[.!?])\s+', cleaned_paragraph)
                    for sentence in sentences:
                        if sentence.strip() and len(sentence.strip()) > 10:
                            result["summary"].append(sentence.strip())
    
    # Extract decisions section
    decisions_match = re.search(r"##\s*DECISIONS\s*([\s\S]*?)(?=##|$)", response)
    if decisions_match:
        decisions_text = decisions_match.group(1).strip()
        # Extract bullet points
        decision_points = re.findall(r"-\s*(.*?)(?=\n-|\n\n|$)", decisions_text)
        for point in decision_points:
            cleaned = clean_text(point.strip())
            if cleaned and "None identified" not in cleaned and len(cleaned) > 5 and not all(c in '[]' for c in cleaned) and "decision" not in cleaned.lower():
                result["decisions"].append(cleaned)
    
    # Extract action items section
    actions_match = re.search(r"##\s*ACTION\s*ITEMS\s*([\s\S]*?)(?=$)", response)
    if actions_match:
        actions_text = actions_match.group(1).strip()
        # Extract bullet points
        action_points = re.findall(r"-\s*(.*?)(?=\n-|\n\n|$)", actions_text)
        for point in action_points:
            cleaned = clean_text(point.strip())
            if cleaned and "None identified" not in cleaned and len(cleaned) > 5 and not all(c in '[]' for c in cleaned) and not "Person]" in cleaned:
                result["action_items"].append(cleaned)
    
    # If we couldn't extract anything using the expected format, try a fallback approach
    if not any(result.values()):
        # Try to find any bullet points in the response
        all_bullet_points = re.findall(r"-\s*(.*?)(?=\n-|\n\n|$)", response)
        
        # Look for sentences that might be summaries
        summary_candidates = [
            clean_text(point) for point in all_bullet_points 
            if len(point) > 10 and 
            not all(c in '[]' for c in point.strip()) and
            "bullet point" not in point.lower() and
            "person name" not in point.lower()
        ]
        
        if summary_candidates:
            # Use the first few bullet points as summary
            result["summary"] = summary_candidates[:3]
        else:
            # Extract the first few sentences as summary
            sentences = re.split(r'(?<=[.!?])\s+', response)
            good_sentences = [
                clean_text(s.strip()) for s in sentences 
                if len(s.strip()) > 15 and 
                not all(c in '[]' for c in s.strip()) and
                "how to" not in s.strip().lower() and
                "example" not in s.strip().lower()
            ]
            if good_sentences:
                result["summary"] = good_sentences[:2]
    
    # Look for decisions in the plain text if none were found
    if not result["decisions"]:
        # Look for sentences containing decision indicators
        decision_indicators = ["approved", "passed", "agreed", "voted", "decided", "adopted", "resolved"]
        sentences = re.split(r'(?<=[.!?])\s+', response)
        for sentence in sentences:
            cleaned = clean_text(sentence)
            if any(indicator in cleaned.lower() for indicator in decision_indicators) and cleaned not in result["decisions"]:
                if len(cleaned) > 10 and "None identified" not in cleaned:
                    result["decisions"].append(cleaned)
    
    # If we still couldn't extract anything, use rule-based extraction on the original transcript
    if not any(result.values()) and hasattr(parse_response, 'current_transcript'):
        # Use rule-based extraction as a last resort
        logger.warning("Couldn't parse LLM response, falling back to rule-based extraction")
        return extract_meeting_info(parse_response.current_transcript)
    
    # Filter out empty items and placeholders
    result["summary"] = [s for s in result["summary"] if s and len(s) > 5 and not s.isspace()]
    result["decisions"] = [d for d in result["decisions"] if d and len(d) > 5 and not d.isspace()]
    result["action_items"] = [a for a in result["action_items"] if a and len(a) > 5 and not a.isspace()]
    
    return result

def process_with_lmstudio(transcript: str, api_url: str = "http://localhost:1234/v1/chat/completions") -> str:
    """
    Process transcript with LM Studio local API.
    """
    prompt = f"""You are an expert meeting analyst who extracts key information from meeting transcripts with precision and clarity. Analyze the following meeting transcript and extract exactly these three elements:

1. SUMMARY (3-5 bullet points):
   * Identify the main topics discussed
   * Focus on substantive content, not procedural remarks
   * Capture the essence of what was discussed, not just who spoke

2. DECISIONS (all that apply):
   * Extract any clear conclusions reached
   * Include formal votes, agreements, or resolutions
   * Note any approved actions, policies, or plans

3. ACTION ITEMS (all that apply):
   * Identify specific tasks assigned to people
   * Include WHO is responsible, WHAT they need to do, and WHEN it's due
   * Look for commitments, assignments, or follow-up tasks

For each section, if no relevant information exists, write "None identified".

Transcript:
```
{transcript}
```

Format your response using EXACTLY this structure:

## SUMMARY
- [First substantive point from the discussion]
- [Second substantive point from the discussion]
- [Additional substantive points as needed]

## DECISIONS
- [First decision made during the meeting]
- [Additional decisions as needed]

## ACTION ITEMS
- [Person name]: [Specific task] by [Deadline if mentioned]
- [Additional action items as needed]
"""

    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.post(
            api_url,
            headers=headers,
            json={
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 1000
            },
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get("choices", [{}])[0].get("message", {}).get("content", "Error: No response content")
        else:
            logger.error(f"LM Studio API error: {response.status_code}")
            return f"Error: LM Studio API returned status code {response.status_code}"
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error: {e}")
        return f"Error: {str(e)}"

def check_lmstudio_available(api_url: str = "http://localhost:1234/v1/models") -> bool:
    """
    Check if LM Studio is available.
    """
    try:
        response = requests.get(api_url, timeout=5)
        if response.status_code == 200:
            logger.info("LM Studio is available.")
            return True
        else:
            logger.warning("LM Studio returned an error.")
            return False
    except requests.exceptions.RequestException:
        logger.warning("LM Studio is not available.")
        return False

def main():
    parser = argparse.ArgumentParser(description='Extract meeting information from transcripts using LLMs')
    parser.add_argument('--input', type=str, default='data/chunked_transcripts.csv', 
                        help='Input CSV file with transcript chunks')
    parser.add_argument('--output', type=str, default='results/meeting_extractions.csv', 
                        help='Output CSV file with extracted information')
    parser.add_argument('--transcript_col', type=str, default='chunk_text', 
                        help='Name of column containing transcript text')
    parser.add_argument('--model', type=str, default='mistral', 
                        help='Ollama model to use (e.g., llama3.2, mistral, gemma)')
    parser.add_argument('--sample', type=int, default=None, 
                        help='Use a sample of chunks for faster processing')
    parser.add_argument('--max_workers', type=int, default=2,
                        help='Maximum number of parallel workers')
    parser.add_argument('--chunk_size', type=int, default=10,
                        help='Number of transcripts to process in each main chunk')
    
    args = parser.parse_args()
    
    # Check if Ollama is available
    if not check_ollama_available():
        logger.error("Ollama is required for this script to run.")
        logger.info("Please install Ollama from https://ollama.com/download")
        logger.info("After installation, pull a model with: ollama pull mistral")
        return
    
    # Load data
    logger.info(f"Loading data from {args.input}...")
    df = pd.read_csv(args.input)
    logger.info(f"Loaded {len(df)} chunks")
    
    # Take a sample if specified
    if args.sample and args.sample < len(df):
        logger.info(f"Using a sample of {args.sample} chunks")
        df = df.sample(args.sample, random_state=42)
    
    # Process transcripts
    process_transcripts(
        df, 
        transcript_col=args.transcript_col,
        llm_type="ollama",
        model_name=args.model,
        max_workers=args.max_workers,
        chunk_size=args.chunk_size,
        output_path=args.output
    )
    
    logger.info("Done!")

if __name__ == "__main__":
    main()
