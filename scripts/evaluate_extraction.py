#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import argparse
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Tuple
from rouge_score import rouge_scorer
from sklearn.metrics import precision_score, recall_score, f1_score
import nltk
from nltk.tokenize import word_tokenize
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from tqdm import tqdm
import requests
import logging
from collections import defaultdict

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Download NLTK resources if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

def load_dataset(file_path: str) -> pd.DataFrame:
    """Load dataset with transcripts and reference summaries."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    df = pd.read_csv(file_path)
    logger.info(f"Loaded dataset with {len(df)} samples")
    return df

def compute_rouge_scores(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute ROUGE scores between predicted and reference summaries."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    scores = {
        'rouge1_precision': [],
        'rouge1_recall': [],
        'rouge1_fmeasure': [],
        'rouge2_precision': [],
        'rouge2_recall': [],
        'rouge2_fmeasure': [],
        'rougeL_precision': [],
        'rougeL_recall': [],
        'rougeL_fmeasure': []
    }
    
    for pred, ref in zip(predictions, references):
        # Skip empty predictions or references
        if not pred or not ref:
            continue
            
        # Calculate scores
        try:
            rouge_scores = scorer.score(ref, pred)
            
            # Add to lists
            scores['rouge1_precision'].append(rouge_scores['rouge1'].precision)
            scores['rouge1_recall'].append(rouge_scores['rouge1'].recall)
            scores['rouge1_fmeasure'].append(rouge_scores['rouge1'].fmeasure)
            
            scores['rouge2_precision'].append(rouge_scores['rouge2'].precision)
            scores['rouge2_recall'].append(rouge_scores['rouge2'].recall)
            scores['rouge2_fmeasure'].append(rouge_scores['rouge2'].fmeasure)
            
            scores['rougeL_precision'].append(rouge_scores['rougeL'].precision)
            scores['rougeL_recall'].append(rouge_scores['rougeL'].recall)
            scores['rougeL_fmeasure'].append(rouge_scores['rougeL'].fmeasure)
        except Exception as e:
            logger.error(f"Error computing ROUGE score: {e}")
    
    # Calculate average scores
    avg_scores = {}
    for metric, values in scores.items():
        if values:  # Check if list is not empty
            avg_scores[metric] = sum(values) / len(values)
        else:
            avg_scores[metric] = 0.0
    
    return avg_scores

def evaluate_action_items(predicted_actions: List[List[str]], 
                         reference_actions: List[List[str]]) -> Dict[str, float]:
    """
    Evaluate action item extraction using precision, recall, and F1 score.
    
    This is a simplified evaluation that checks if each predicted action item
    contains the key elements from the reference action items.
    """
    results = {
        'precision': [],
        'recall': [],
        'f1': []
    }
    
    for pred_list, ref_list in zip(predicted_actions, reference_actions):
        # Skip if either list is empty
        if not pred_list or not ref_list:
            continue
        
        # Tokenize each action item
        pred_tokens = [set(word_tokenize(item.lower())) for item in pred_list]
        ref_tokens = [set(word_tokenize(item.lower())) for item in ref_list]
        
        # Calculate matches
        true_positives = 0
        for pred_set in pred_tokens:
            # Check if this prediction matches any reference
            for ref_set in ref_tokens:
                # If there's significant overlap, count as a match
                overlap = len(pred_set.intersection(ref_set))
                if overlap >= min(3, len(ref_set) // 2):  # At least 3 words or half of reference
                    true_positives += 1
                    break
        
        # Calculate metrics
        precision = true_positives / len(pred_tokens) if pred_tokens else 0
        recall = true_positives / len(ref_tokens) if ref_tokens else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        results['precision'].append(precision)
        results['recall'].append(recall)
        results['f1'].append(f1)
    
    # Calculate averages
    avg_results = {}
    for metric, values in results.items():
        if values:  # Check if list is not empty
            avg_results[metric] = sum(values) / len(values)
        else:
            avg_results[metric] = 0.0
    
    return avg_results

def generate_t5_summaries(transcripts: List[str], model_name: str = "t5-small") -> List[str]:
    """Generate summaries using T5 model as a baseline."""
    logger.info(f"Generating summaries with {model_name}...")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    summaries = []
    
    for transcript in tqdm(transcripts, desc="Generating T5 summaries"):
        # Truncate long transcripts to fit model's max length
        max_length = 512  # T5-small has a limit
        input_text = "summarize: " + transcript[:10000]  # Add prefix for T5
        
        # Tokenize and generate
        inputs = tokenizer(input_text, return_tensors="pt", max_length=max_length, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate summary
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_length=150,
                num_beams=4,
                early_stopping=True
            )
        
        # Decode and append
        summary = tokenizer.decode(output[0], skip_special_tokens=True)
        summaries.append(summary)
    
    return summaries

def analyze_with_api(transcripts: List[str], model: str = "mistral") -> List[Dict[str, Any]]:
    """Process transcripts using the FastAPI backend."""
    logger.info(f"Processing {len(transcripts)} transcripts with API using {model} model...")
    
    results = []
    
    for transcript in tqdm(transcripts, desc="API Processing"):
        try:
            response = requests.post(
                "http://localhost:8000/analyze",
                json={
                    "text": transcript,
                    "model": model,
                    "max_chunk_size": 1500,
                    "use_cache": True
                },
                timeout=60
            )
            
            if response.status_code == 200:
                results.append(response.json())
            else:
                logger.error(f"API error: {response.status_code}")
                results.append(None)
        except Exception as e:
            logger.error(f"Error calling API: {e}")
            results.append(None)
    
    return results

def plot_rouge_comparison(baseline_scores: Dict[str, float], 
                         ollama_scores: Dict[str, float], 
                         output_path: str = "results/rouge_comparison.png"):
    """Plot ROUGE score comparison between baseline and Ollama models."""
    # Prepare data
    metrics = ['rouge1_fmeasure', 'rouge2_fmeasure', 'rougeL_fmeasure']
    labels = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']
    
    baseline_values = [baseline_scores[m] for m in metrics]
    ollama_values = [ollama_scores[m] for m in metrics]
    
    # Create plot
    plt.figure(figsize=(10, 6))
    
    x = np.arange(len(labels))
    width = 0.35
    
    plt.bar(x - width/2, baseline_values, width, label='T5-small (Baseline)')
    plt.bar(x + width/2, ollama_values, width, label='Ollama (Mistral)')
    
    plt.xlabel('Metric')
    plt.ylabel('Score')
    plt.title('ROUGE Score Comparison')
    plt.xticks(x, labels)
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save plot
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved ROUGE comparison plot to {output_path}")
    
    # Show improvement percentages
    improvements = {}
    for i, metric in enumerate(metrics):
        if baseline_values[i] > 0:
            imp_percent = (ollama_values[i] - baseline_values[i]) / baseline_values[i] * 100
            improvements[labels[i]] = imp_percent
    
    return improvements

def plot_action_item_metrics(baseline_metrics: Dict[str, float], 
                            ollama_metrics: Dict[str, float],
                            output_path: str = "results/action_item_metrics.png"):
    """Plot action item extraction metrics comparison."""
    # Prepare data
    metrics = ['precision', 'recall', 'f1']
    labels = ['Precision', 'Recall', 'F1 Score']
    
    baseline_values = [baseline_metrics[m] for m in metrics]
    ollama_values = [ollama_metrics[m] for m in metrics]
    
    # Create plot
    plt.figure(figsize=(10, 6))
    
    x = np.arange(len(labels))
    width = 0.35
    
    plt.bar(x - width/2, baseline_values, width, label='Baseline (Rule-based)')
    plt.bar(x + width/2, ollama_values, width, label='Ollama (Mistral)')
    
    plt.xlabel('Metric')
    plt.ylabel('Score')
    plt.title('Action Item Extraction Metrics')
    plt.xticks(x, labels)
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save plot
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved action item metrics plot to {output_path}")
    
    # Show improvement percentages
    improvements = {}
    for i, metric in enumerate(metrics):
        if baseline_values[i] > 0:
            imp_percent = (ollama_values[i] - baseline_values[i]) / baseline_values[i] * 100
            improvements[labels[i]] = imp_percent
    
    return improvements

def main():
    parser = argparse.ArgumentParser(description='Evaluate meeting transcript extraction quality')
    parser.add_argument('--dataset', type=str, default='data/meeting_summaries_clean.csv', 
                        help='Path to dataset with transcripts and reference summaries')
    parser.add_argument('--transcript_col', type=str, default='transcript', 
                        help='Column name containing transcript text')
    parser.add_argument('--summary_col', type=str, default='summary', 
                        help='Column name containing reference summaries')
    parser.add_argument('--action_items_col', type=str, default='action_items', 
                        help='Column name containing reference action items')
    parser.add_argument('--sample', type=int, default=10, 
                        help='Number of samples to evaluate (for faster testing)')
    parser.add_argument('--model', type=str, default='mistral', 
                        help='Ollama model to use')
    parser.add_argument('--output_dir', type=str, default='results', 
                        help='Directory to save evaluation results')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    df = load_dataset(args.dataset)
    
    # Take a sample if specified
    if args.sample and args.sample < len(df):
        df = df.sample(args.sample, random_state=42)
    
    # Extract data
    transcripts = df[args.transcript_col].tolist()
    reference_summaries = df[args.summary_col].tolist()
    
    # Extract reference action items (if available)
    reference_action_items = []
    if args.action_items_col in df.columns:
        for items in df[args.action_items_col]:
            if isinstance(items, str):
                try:
                    # Try to parse as JSON list
                    action_items = json.loads(items)
                    reference_action_items.append(action_items)
                except:
                    # If not JSON, split by newlines
                    action_items = [item.strip() for item in items.split('\n') if item.strip()]
                    reference_action_items.append(action_items)
            else:
                reference_action_items.append([])
    else:
        # If no action items column, use empty lists
        reference_action_items = [[] for _ in range(len(transcripts))]
    
    # Generate baseline summaries with T5
    baseline_summaries = generate_t5_summaries(transcripts)
    
    # Process with API
    api_results = analyze_with_api(transcripts, model=args.model)
    
    # Extract summaries and action items from API results
    ollama_summaries = []
    ollama_action_items = []
    
    for result in api_results:
        if result and result.get('success', False):
            # Join summary points into a single string
            summary = ' '.join(result['results']['summary'])
            ollama_summaries.append(summary)
            
            # Get action items as a list
            action_items = result['results']['action_items']
            ollama_action_items.append(action_items)
        else:
            ollama_summaries.append('')
            ollama_action_items.append([])
    
    # Compute ROUGE scores
    baseline_rouge = compute_rouge_scores(baseline_summaries, reference_summaries)
    ollama_rouge = compute_rouge_scores(ollama_summaries, reference_summaries)
    
    # Evaluate action item extraction
    # For baseline, use rule-based extraction from extract_meeting_info.py
    from extract_meeting_info import extract_meeting_info
    
    baseline_action_items = []
    for transcript in transcripts:
        extracted = extract_meeting_info(transcript)
        baseline_action_items.append(extracted['action_items'])
    
    baseline_action_metrics = evaluate_action_items(baseline_action_items, reference_action_items)
    ollama_action_metrics = evaluate_action_items(ollama_action_items, reference_action_items)
    
    # Plot results
    rouge_improvements = plot_rouge_comparison(baseline_rouge, ollama_rouge, 
                                              os.path.join(args.output_dir, 'rouge_comparison.png'))
    
    action_improvements = plot_action_item_metrics(baseline_action_metrics, ollama_action_metrics,
                                                  os.path.join(args.output_dir, 'action_item_metrics.png'))
    
    # Save detailed results
    results = {
        'baseline_rouge': baseline_rouge,
        'ollama_rouge': ollama_rouge,
        'rouge_improvements': rouge_improvements,
        'baseline_action_metrics': baseline_action_metrics,
        'ollama_action_metrics': ollama_action_metrics,
        'action_improvements': action_improvements
    }
    
    with open(os.path.join(args.output_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    logger.info("\n=== Evaluation Results ===")
    
    logger.info("\nROUGE Scores:")
    logger.info(f"  Baseline (T5-small):")
    logger.info(f"    ROUGE-1 F1: {baseline_rouge['rouge1_fmeasure']:.4f}")
    logger.info(f"    ROUGE-2 F1: {baseline_rouge['rouge2_fmeasure']:.4f}")
    logger.info(f"    ROUGE-L F1: {baseline_rouge['rougeL_fmeasure']:.4f}")
    
    logger.info(f"  Ollama ({args.model}):")
    logger.info(f"    ROUGE-1 F1: {ollama_rouge['rouge1_fmeasure']:.4f}")
    logger.info(f"    ROUGE-2 F1: {ollama_rouge['rouge2_fmeasure']:.4f}")
    logger.info(f"    ROUGE-L F1: {ollama_rouge['rougeL_fmeasure']:.4f}")
    
    logger.info("\nROUGE Improvements:")
    for metric, improvement in rouge_improvements.items():
        logger.info(f"  {metric}: {improvement:.1f}%")
    
    logger.info("\nAction Item Extraction:")
    logger.info(f"  Baseline (Rule-based):")
    logger.info(f"    Precision: {baseline_action_metrics['precision']:.4f}")
    logger.info(f"    Recall: {baseline_action_metrics['recall']:.4f}")
    logger.info(f"    F1 Score: {baseline_action_metrics['f1']:.4f}")
    
    logger.info(f"  Ollama ({args.model}):")
    logger.info(f"    Precision: {ollama_action_metrics['precision']:.4f}")
    logger.info(f"    Recall: {ollama_action_metrics['recall']:.4f}")
    logger.info(f"    F1 Score: {ollama_action_metrics['f1']:.4f}")
    
    logger.info("\nAction Item Extraction Improvements:")
    for metric, improvement in action_improvements.items():
        logger.info(f"  {metric}: {improvement:.1f}%")
    
    logger.info(f"\nDetailed results saved to {os.path.join(args.output_dir, 'evaluation_results.json')}")

if __name__ == "__main__":
    main()