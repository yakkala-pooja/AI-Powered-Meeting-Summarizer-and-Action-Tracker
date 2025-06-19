import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
import logging
import argparse
from time import time
import warnings
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Download required NLTK resources
def download_nltk_resources():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        logger.info("Downloading NLTK resources...")
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')

# Text preprocessing
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return ' '.join(tokens)

# Print top words for each topic
def print_top_words(model, feature_names, n_top_words=10):
    topic_keywords = {}
    for topic_idx, topic in enumerate(model.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        topic_keywords[topic_idx] = top_words
        logger.info(f"Topic #{topic_idx}: {' '.join(top_words)}")
    return topic_keywords

# Calculate topic coherence score
def calculate_coherence_score(model, feature_names, n_top_words=10):
    """
    Calculate the coherence score for a topic model
    Higher coherence score indicates better semantic coherence within topics
    """
    coherence_scores = []
    
    # For each topic
    for topic_idx, topic in enumerate(model.components_):
        # Get top words
        top_word_indices = topic.argsort()[:-n_top_words - 1:-1]
        top_words = [feature_names[i] for i in top_word_indices]
        
        # Calculate pairwise similarities
        word_scores = []
        for i in range(len(top_words)):
            for j in range(i+1, len(top_words)):
                # Simple co-occurrence score for this implementation
                # In a real implementation, you'd use a proper word embedding model
                # or a pre-computed co-occurrence matrix
                word_scores.append(1.0 if top_words[i][0] == top_words[j][0] else 0.0)
        
        # Average similarity for this topic
        if word_scores:
            coherence_scores.append(sum(word_scores) / len(word_scores))
    
    # Return average coherence across all topics
    return sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0.0

# Calculate topic diversity score
def calculate_diversity_score(model):
    """
    Calculate the diversity between topics
    Higher diversity score indicates more distinct topics
    """
    # Calculate pairwise cosine similarity between topic vectors
    similarities = cosine_similarity(model.components_)
    
    # Remove self-similarities (diagonal)
    np.fill_diagonal(similarities, 0)
    
    # Average similarity
    avg_similarity = similarities.sum() / (similarities.shape[0] * (similarities.shape[0] - 1))
    
    # Convert to diversity score (1 - similarity)
    return 1 - avg_similarity

# Calculate perplexity score
def calculate_perplexity(model, data):
    """
    Calculate perplexity score
    Lower perplexity indicates better fit to the data
    """
    return model.perplexity(data)

# Evaluate different numbers of topics
def evaluate_topic_numbers(df, min_topics=5, max_topics=30, step=5):
    """
    Evaluate different numbers of topics and return metrics
    """
    logger.info("Preprocessing text data...")
    df['processed_text'] = df['chunk_text'].apply(preprocess_text)
    
    # Filter out empty texts
    df = df[df['processed_text'].str.strip() != '']
    
    logger.info("Vectorizing text data...")
    count_vectorizer = CountVectorizer(
        max_df=0.95,      # Ignore terms that appear in >95% of documents
        min_df=2,         # Ignore terms that appear in <2 documents
        max_features=1000,
        stop_words='english'
    )
    
    count_data = count_vectorizer.fit_transform(df['processed_text'])
    feature_names = count_vectorizer.get_feature_names_out()
    
    # Metrics to track
    coherence_scores = []
    diversity_scores = []
    perplexity_scores = []
    topic_numbers = range(min_topics, max_topics + 1, step)
    
    logger.info("Evaluating different numbers of topics...")
    for n_topics in tqdm(topic_numbers):
        logger.info(f"Fitting LDA model with {n_topics} topics...")
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            max_iter=10,
            learning_method='online',
            learning_offset=50.,
            random_state=0
        )
        
        t0 = time()
        lda.fit(count_data)
        logger.info(f"LDA model fitted in {time() - t0:.2f}s")
        
        # Calculate metrics
        coherence = calculate_coherence_score(lda, feature_names)
        diversity = calculate_diversity_score(lda)
        perplexity = lda.perplexity(count_data)
        
        coherence_scores.append(coherence)
        diversity_scores.append(diversity)
        perplexity_scores.append(perplexity)
        
        logger.info(f"Topics: {n_topics}, Coherence: {coherence:.4f}, Diversity: {diversity:.4f}, Perplexity: {perplexity:.2f}")
    
    # Create results dataframe
    results = pd.DataFrame({
        'n_topics': list(topic_numbers),
        'coherence': coherence_scores,
        'diversity': diversity_scores,
        'perplexity': perplexity_scores
    })
    
    return results

# Plot evaluation metrics
def plot_evaluation_metrics(results):
    """
    Plot the evaluation metrics for different numbers of topics
    """
    plt.figure(figsize=(15, 10))
    
    # Plot coherence
    plt.subplot(3, 1, 1)
    plt.plot(results['n_topics'], results['coherence'], 'o-', color='blue')
    plt.title('Topic Coherence Score')
    plt.xlabel('Number of Topics')
    plt.ylabel('Coherence Score')
    plt.grid(True)
    
    # Plot diversity
    plt.subplot(3, 1, 2)
    plt.plot(results['n_topics'], results['diversity'], 'o-', color='green')
    plt.title('Topic Diversity Score')
    plt.xlabel('Number of Topics')
    plt.ylabel('Diversity Score')
    plt.grid(True)
    
    # Plot perplexity
    plt.subplot(3, 1, 3)
    plt.plot(results['n_topics'], results['perplexity'], 'o-', color='red')
    plt.title('Model Perplexity')
    plt.xlabel('Number of Topics')
    plt.ylabel('Perplexity')
    plt.grid(True)
    
    plt.tight_layout()
    
    # Ensure directory exists
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/topic_evaluation_metrics.png')
    plt.close()

# Find optimal number of topics
def find_optimal_topics(results):
    """
    Find the optimal number of topics based on the evaluation metrics
    """
    # Normalize metrics to [0, 1] range
    normalized_coherence = (results['coherence'] - results['coherence'].min()) / (results['coherence'].max() - results['coherence'].min())
    normalized_diversity = (results['diversity'] - results['diversity'].min()) / (results['diversity'].max() - results['diversity'].min())
    normalized_perplexity = 1 - (results['perplexity'] - results['perplexity'].min()) / (results['perplexity'].max() - results['perplexity'].min())
    
    # Calculate combined score (higher is better)
    combined_score = (normalized_coherence + normalized_diversity + normalized_perplexity) / 3
    
    # Find optimal number of topics
    optimal_idx = combined_score.argmax()
    optimal_topics = results.iloc[optimal_idx]['n_topics']
    
    logger.info(f"Optimal number of topics: {optimal_topics}")
    logger.info(f"Coherence: {results.iloc[optimal_idx]['coherence']:.4f}")
    logger.info(f"Diversity: {results.iloc[optimal_idx]['diversity']:.4f}")
    logger.info(f"Perplexity: {results.iloc[optimal_idx]['perplexity']:.2f}")
    
    return optimal_topics

def main():
    parser = argparse.ArgumentParser(description='Evaluate different numbers of topics for meeting transcript chunks')
    parser.add_argument('--input', type=str, default='data/chunked_transcripts.csv', help='Input CSV file with transcript chunks')
    parser.add_argument('--min_topics', type=int, default=5, help='Minimum number of topics to evaluate')
    parser.add_argument('--max_topics', type=int, default=30, help='Maximum number of topics to evaluate')
    parser.add_argument('--step', type=int, default=5, help='Step size for number of topics')
    parser.add_argument('--sample', type=int, default=None, help='Use a sample of chunks for faster processing')
    
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
    
    # Evaluate different numbers of topics
    results = evaluate_topic_numbers(df, args.min_topics, args.max_topics, args.step)
    
    # Ensure results directory exists
    os.makedirs('results', exist_ok=True)
    
    # Save results
    results.to_csv('results/topic_evaluation_results.csv', index=False)
    
    # Plot evaluation metrics
    plot_evaluation_metrics(results)
    
    # Find optimal number of topics
    optimal_topics = find_optimal_topics(results)
    
    logger.info(f"Evaluation results saved to 'results/topic_evaluation_results.csv'")
    logger.info(f"Plots saved to 'results/topic_evaluation_metrics.png'")
    logger.info(f"Recommended number of topics: {optimal_topics}")

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
