import nltk
import spacy
import pandas as pd
import argparse
from tqdm import tqdm
import os
import logging
from typing import List, Generator, Dict, Any, Union, Tuple

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TranscriptChunker:
    """A class to split transcripts into manageable chunks for LLM processing."""
    
    def __init__(self, use_spacy: bool = False, sentences_per_chunk: int = 6):
        """
        Initialize the chunker.
        
        Args:
            use_spacy: Whether to use spaCy (True) or NLTK (False) for sentence splitting
            sentences_per_chunk: Number of sentences to include in each chunk (default: 6)
        """
        self.use_spacy = use_spacy
        self.sentences_per_chunk = sentences_per_chunk
        
        if use_spacy:
            logger.info("Loading spaCy model...")
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.info("Downloading spaCy model...")
                os.system("python -m spacy download en_core_web_sm")
                self.nlp = spacy.load("en_core_web_sm")
        else:
            logger.info("Downloading NLTK punkt tokenizer...")
            nltk.download('punkt', quiet=True)
            self.sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    
    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text: The input text to split
            
        Returns:
            List of sentences
        """
        if not text or not isinstance(text, str):
            return []
        
        if self.use_spacy:
            # Process text in chunks to avoid memory issues with very long texts
            max_length = 100000  # Process 100k characters at a time
            if len(text) > max_length:
                sentences = []
                for i in range(0, len(text), max_length):
                    chunk = text[i:i + max_length]
                    # Make sure we don't cut in the middle of a sentence
                    if i + max_length < len(text):
                        # Find the last period, question mark, or exclamation point
                        last_end = max(chunk.rfind('.'), chunk.rfind('?'), chunk.rfind('!'))
                        if last_end > 0:
                            chunk = chunk[:last_end + 1]
                    doc = self.nlp(chunk)
                    sentences.extend([sent.text.strip() for sent in doc.sents])
                return sentences
            else:
                doc = self.nlp(text)
                return [sent.text.strip() for sent in doc.sents]
        else:
            return self.sent_tokenizer.tokenize(text)
    
    def create_chunks(self, sentences: List[str]) -> List[str]:
        """
        Group sentences into chunks.
        
        Args:
            sentences: List of sentences to group
            
        Returns:
            List of text chunks
        """
        chunks = []
        
        for i in range(0, len(sentences), self.sentences_per_chunk):
            chunk = ' '.join(sentences[i:i + self.sentences_per_chunk])
            chunks.append(chunk)
        
        return chunks
    
    def process_transcript(self, transcript: str) -> List[str]:
        """
        Process a transcript: split into sentences and then into chunks.
        
        Args:
            transcript: The meeting transcript text
            
        Returns:
            List of transcript chunks
        """
        sentences = self.split_into_sentences(transcript)
        logger.info(f"Split transcript into {len(sentences)} sentences")
        
        chunks = self.create_chunks(sentences)
        logger.info(f"Created {len(chunks)} chunks")
        
        return chunks
    
    def process_dataset(self, df: pd.DataFrame, transcript_col: str = 'meeting_transcript') -> List[Dict[str, Any]]:
        """
        Process a dataset of transcripts.
        
        Args:
            df: DataFrame containing transcripts
            transcript_col: Name of the column containing transcript text
            
        Returns:
            List of dictionaries with transcript ID, chunk ID, and chunk text
        """
        all_chunks = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing transcripts"):
            transcript = row[transcript_col]
            chunks = self.process_transcript(transcript)
            
            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append({
                    'transcript_id': idx,
                    'chunk_id': chunk_idx,
                    'chunk_text': chunk,
                    'original_summary': row.get('summary', '')
                })
        
        return all_chunks

def process_file(file_path: str, output_path: str, use_spacy: bool = False, 
                sentences_per_chunk: int = 6, transcript_col: str = 'meeting_transcript'):
    """
    Process a file containing transcripts.
    
    Args:
        file_path: Path to the input file (CSV)
        output_path: Path to save the output file
        use_spacy: Whether to use spaCy for sentence splitting
        sentences_per_chunk: Number of sentences per chunk
        transcript_col: Name of the column containing transcript text
    """
    logger.info(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)
    
    logger.info(f"Loaded {len(df)} transcripts")
    
    chunker = TranscriptChunker(use_spacy=use_spacy, sentences_per_chunk=sentences_per_chunk)
    chunks = chunker.process_dataset(df, transcript_col=transcript_col)
    
    chunks_df = pd.DataFrame(chunks)
    chunks_df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(chunks_df)} chunks to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Split transcripts into chunks for LLM processing")
    parser.add_argument("--input", type=str, default="data/meeting_summaries_clean.csv", help="Path to input CSV file")
    parser.add_argument("--output", type=str, default="data/chunked_transcripts.csv", help="Path to output CSV file")
    parser.add_argument("--use_spacy", action="store_true", help="Use spaCy for sentence splitting (default: NLTK)")
    parser.add_argument("--sentences_per_chunk", type=int, default=6, help="Number of sentences per chunk (default: 6)")
    parser.add_argument("--transcript_col", type=str, default="meeting_transcript", help="Name of transcript column")
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    process_file(
        args.input, 
        args.output, 
        args.use_spacy, 
        args.sentences_per_chunk,
        args.transcript_col
    )

if __name__ == "__main__":
    main() 