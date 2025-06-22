# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import nltk
import re
import matplotlib.pyplot as plt
import argparse
import os
from tqdm import tqdm
import logging
from typing import List, Dict, Tuple, Any, Optional
from nltk.tokenize import sent_tokenize

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import sentiment analysis tools
try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    HAVE_VADER = True
    logger.info("VADER sentiment analyzer available")
except ImportError:
    HAVE_VADER = False
    logger.warning("VADER not available, will try TextBlob")

try:
    from textblob import TextBlob
    HAVE_TEXTBLOB = True
    logger.info("TextBlob available for sentiment analysis")
except ImportError:
    HAVE_TEXTBLOB = False
    logger.warning("TextBlob not available")

# Download required NLTK resources
def download_nltk_resources():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('sentiment/vader_lexicon')
    except LookupError:
        logger.info("Downloading NLTK resources...")
        nltk.download('punkt')
        nltk.download('vader_lexicon')

# Analyze sentiment using VADER
def analyze_sentiment_vader(text: str) -> Dict[str, float]:
    """
    Analyze sentiment of text using VADER.
    
    Args:
        text: Input text
        
    Returns:
        Dictionary with sentiment scores (neg, neu, pos, compound)
    """
    if not HAVE_VADER:
        raise ImportError("VADER is not available. Please install nltk and download vader_lexicon.")
    
    sid = SentimentIntensityAnalyzer()
    return sid.polarity_scores(text)

# Analyze sentiment using TextBlob
def analyze_sentiment_textblob(text: str) -> Dict[str, float]:
    """
    Analyze sentiment of text using TextBlob.
    
    Args:
        text: Input text
        
    Returns:
        Dictionary with sentiment scores (polarity, subjectivity)
    """
    if not HAVE_TEXTBLOB:
        raise ImportError("TextBlob is not available. Please install textblob.")
    
    blob = TextBlob(text)
    return {
        'polarity': blob.sentiment.polarity,
        'subjectivity': blob.sentiment.subjectivity,
        'compound': blob.sentiment.polarity,  # For compatibility with VADER
        'pos': max(0, blob.sentiment.polarity),  # Approximate positive score
        'neg': max(0, -blob.sentiment.polarity),  # Approximate negative score
        'neu': 1 - abs(blob.sentiment.polarity)  # Approximate neutral score
    }

# Analyze sentiment using available tools
def analyze_sentiment(text: str) -> Dict[str, float]:
    """
    Analyze sentiment using the best available tool.
    
    Args:
        text: Input text
        
    Returns:
        Dictionary with sentiment scores
    """
    if HAVE_VADER:
        return analyze_sentiment_vader(text)
    elif HAVE_TEXTBLOB:
        return analyze_sentiment_textblob(text)
    else:
        raise ImportError("No sentiment analysis tools available. Please install nltk or textblob.")

# Split text into sentences
def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences.
    
    Args:
        text: Input text
        
    Returns:
        List of sentences
    """
    if not text or not isinstance(text, str):
        return []
    
    return sent_tokenize(text)

# Detect significant sentiment shifts
def detect_sentiment_shifts(sentences: List[str], threshold: float = 0.5) -> List[int]:
    """
    Detect significant shifts in sentiment.
    
    Args:
        sentences: List of sentences
        threshold: Threshold for significant sentiment shift (0.0-1.0)
        
    Returns:
        List of indices where sentiment shifts significantly
    """
    if not sentences:
        return []
    
    # Analyze sentiment for each sentence
    sentiments = []
    for sentence in sentences:
        sentiment = analyze_sentiment(sentence)
        sentiments.append(sentiment)
    
    # Detect shifts in compound sentiment
    shift_indices = []
    prev_compound = sentiments[0]['compound']
    
    for i, sentiment in enumerate(sentiments[1:], 1):
        current_compound = sentiment['compound']
        # Calculate absolute difference in sentiment
        shift = abs(current_compound - prev_compound)
        
        # If shift exceeds threshold, mark as a significant shift
        if shift >= threshold:
            shift_indices.append(i)
        
        prev_compound = current_compound
    
    return shift_indices

# Segment text based on sentiment shifts
def segment_by_sentiment(text: str, threshold: float = 0.5) -> List[Dict[str, Any]]:
    """
    Segment text based on significant sentiment shifts.
    
    Args:
        text: Input text
        threshold: Threshold for significant sentiment shift (0.0-1.0)
        
    Returns:
        List of segments with text and sentiment information
    """
    # Split into sentences
    sentences = split_into_sentences(text)
    if not sentences:
        return []
    
    # Detect sentiment shifts
    shift_indices = detect_sentiment_shifts(sentences, threshold)
    
    # Create segments
    segments = []
    start_idx = 0
    
    for shift_idx in shift_indices:
        # Get segment text
        segment_sentences = sentences[start_idx:shift_idx]
        segment_text = ' '.join(segment_sentences)
        
        # Analyze segment sentiment
        segment_sentiment = analyze_sentiment(segment_text)
        
        # Add segment
        segments.append({
            'text': segment_text,
            'sentiment': segment_sentiment,
            'start_sentence': start_idx,
            'end_sentence': shift_idx - 1,
            'num_sentences': len(segment_sentences)
        })
        
        start_idx = shift_idx
    
    # Add final segment
    final_sentences = sentences[start_idx:]
    if final_sentences:
        final_text = ' '.join(final_sentences)
        final_sentiment = analyze_sentiment(final_text)
        
        segments.append({
            'text': final_text,
            'sentiment': final_sentiment,
            'start_sentence': start_idx,
            'end_sentence': len(sentences) - 1,
            'num_sentences': len(final_sentences)
        })
    
    return segments

# Process a dataset of transcripts
def process_transcripts(df: pd.DataFrame, 
                        transcript_col: str = 'chunk_text', 
                        threshold: float = 0.5) -> pd.DataFrame:
    """
    Process a dataset of transcripts and segment by sentiment.
    
    Args:
        df: DataFrame containing transcripts
        transcript_col: Name of column containing transcript text
        threshold: Threshold for significant sentiment shift
        
    Returns:
        DataFrame with sentiment segments
    """
    all_segments = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing transcripts"):
        transcript = row[transcript_col]
        
        # Skip empty transcripts
        if not isinstance(transcript, str) or not transcript.strip():
            continue
        
        # Segment by sentiment
        segments = segment_by_sentiment(transcript, threshold)
        
        # Add segments to result
        for segment_idx, segment in enumerate(segments):
            all_segments.append({
                'transcript_id': idx,
                'segment_id': segment_idx,
                'segment_text': segment['text'],
                'sentiment_compound': segment['sentiment']['compound'],
                'sentiment_positive': segment['sentiment'].get('pos', 0),
                'sentiment_negative': segment['sentiment'].get('neg', 0),
                'sentiment_neutral': segment['sentiment'].get('neu', 0),
                'num_sentences': segment['num_sentences'],
                'original_chunk': transcript
            })
    
    return pd.DataFrame(all_segments)

# Visualize sentiment distribution
def visualize_sentiment(segments_df: pd.DataFrame, output_file: str = None):
    """
    Visualize sentiment distribution in segments.
    
    Args:
        segments_df: DataFrame with sentiment segments
        output_file: Path to save visualization
    """
    plt.figure(figsize=(12, 8))
    
    # Histogram of compound sentiment
    plt.subplot(2, 2, 1)
    plt.hist(segments_df['sentiment_compound'], bins=20, color='blue', alpha=0.7)
    plt.title('Distribution of Compound Sentiment')
    plt.xlabel('Compound Sentiment Score')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    
    # Histogram of segment lengths
    plt.subplot(2, 2, 2)
    plt.hist(segments_df['num_sentences'], bins=20, color='green', alpha=0.7)
    plt.title('Distribution of Segment Lengths')
    plt.xlabel('Number of Sentences')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    
    # Scatter plot of sentiment vs segment length
    plt.subplot(2, 2, 3)
    plt.scatter(segments_df['sentiment_compound'], segments_df['num_sentences'], 
                alpha=0.5, c=segments_df['sentiment_compound'], cmap='coolwarm')
    plt.colorbar(label='Sentiment')
    plt.title('Sentiment vs Segment Length')
    plt.xlabel('Compound Sentiment Score')
    plt.ylabel('Number of Sentences')
    plt.grid(True, alpha=0.3)
    
    # Stacked bar for positive/negative/neutral distribution
    plt.subplot(2, 2, 4)
    sentiment_counts = segments_df[['sentiment_positive', 'sentiment_negative', 'sentiment_neutral']].mean()
    sentiment_counts.plot(kind='bar', color=['green', 'red', 'gray'])
    plt.title('Average Sentiment Distribution')
    plt.ylabel('Average Score')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file)
        logger.info(f"Saved visualization to {output_file}")
    
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Segment meeting transcripts based on sentiment shifts')
    parser.add_argument('--input', type=str, default='data/chunked_transcripts.csv', 
                        help='Input CSV file with transcript chunks')
    parser.add_argument('--output', type=str, default='results/sentiment_segments.csv', 
                        help='Output CSV file with sentiment segments')
    parser.add_argument('--threshold', type=float, default=0.5, 
                        help='Threshold for significant sentiment shift (0.0-1.0)')
    parser.add_argument('--transcript_col', type=str, default='chunk_text', 
                        help='Name of column containing transcript text')
    parser.add_argument('--sample', type=int, default=None, 
                        help='Use a sample of chunks for faster processing')
    parser.add_argument('--visualize', action='store_true', 
                        help='Generate visualizations of sentiment distribution')
    
    args = parser.parse_args()
    
    # Download NLTK resources
    download_nltk_resources()
    
    # Load data
    logger.info(f"Loading data from {args.input}...")
    df = pd.read_csv(args.input)
    logger.info(f"Loaded {len(df)} chunks")
    
    # Take a sample if specified
    if args.sample and args.sample < len(df):
        logger.info(f"Using a sample of {args.sample} chunks")
        df = df.sample(args.sample, random_state=42)
    
    # Process transcripts
    segments_df = process_transcripts(
        df, 
        transcript_col=args.transcript_col,
        threshold=args.threshold
    )
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Save results
    logger.info(f"Saving {len(segments_df)} sentiment segments to {args.output}...")
    segments_df.to_csv(args.output, index=False)
    
    # Generate visualizations if requested
    if args.visualize:
        visualization_path = args.output.replace('.csv', '_visualization.png')
        visualize_sentiment(segments_df, visualization_path)
    
    # Print summary statistics
    logger.info("\nSummary Statistics:")
    logger.info(f"Total transcripts processed: {df.shape[0]}")
    logger.info(f"Total sentiment segments: {segments_df.shape[0]}")
    logger.info(f"Average segments per transcript: {segments_df.shape[0] / df.shape[0]:.2f}")
    
    # Show sentiment distribution
    sentiment_distribution = segments_df['sentiment_compound'].describe()
    logger.info("\nSentiment Distribution:")
    logger.info(sentiment_distribution)
    
    # Count positive, negative, and neutral segments
    positive_segments = (segments_df['sentiment_compound'] > 0.05).sum()
    negative_segments = (segments_df['sentiment_compound'] < -0.05).sum()
    neutral_segments = ((segments_df['sentiment_compound'] >= -0.05) & 
                        (segments_df['sentiment_compound'] <= 0.05)).sum()
    
    logger.info(f"\nPositive segments: {positive_segments} ({positive_segments/len(segments_df)*100:.1f}%)")
    logger.info(f"Negative segments: {negative_segments} ({negative_segments/len(segments_df)*100:.1f}%)")
    logger.info(f"Neutral segments: {neutral_segments} ({neutral_segments/len(segments_df)*100:.1f}%)")
    
    logger.info("Done!")

if __name__ == "__main__":
    main() 