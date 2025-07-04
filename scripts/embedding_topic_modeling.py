# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import nltk
import re
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from time import time
import logging
import argparse
from tqdm import tqdm
import warnings
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

# Try to import sentence-transformers for embedding-based topic modeling
try:
    from sentence_transformers import SentenceTransformer
    HAVE_SENTENCE_TRANSFORMERS = True
    logger.info("Sentence Transformers available for embedding-based topic modeling")
except ImportError:
    HAVE_SENTENCE_TRANSFORMERS = False
    logger.warning("Sentence Transformers not available, will use TF-IDF or Count vectorization")

# Try to import BERTopic
try:
    from bertopic import BERTopic
    HAVE_BERTOPIC = True
    logger.info("BERTopic available for advanced topic modeling")
except ImportError:
    HAVE_BERTOPIC = False
    logger.warning("BERTopic not available, will use alternative methods")

# Embedding-based topic modeling using sentence-transformers and clustering
def perform_embedding_topic_modeling(df, n_topics=10, n_top_words=10):
    if not HAVE_SENTENCE_TRANSFORMERS:
        logger.warning("Sentence Transformers not available, falling back to TF-IDF LDA")
        return perform_lda_topic_modeling(df, n_topics, n_top_words)
    
    logger.info("Preprocessing text data...")
    df['processed_text'] = df['chunk_text'].apply(preprocess_text)
    
    # Filter out empty texts
    df = df[df['processed_text'].str.strip() != '']
    
    # Load pre-trained model
    logger.info("Loading sentence transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Generate embeddings
    logger.info("Generating embeddings for chunks...")
    embeddings = model.encode(df['chunk_text'].tolist(), show_progress_bar=True)
    
    # Dimensionality reduction (optional)
    logger.info("Reducing dimensionality with TruncatedSVD...")
    svd = TruncatedSVD(n_components=100, random_state=42)
    reduced_embeddings = svd.fit_transform(embeddings)
    
    # Cluster the embeddings
    logger.info(f"Clustering embeddings into {n_topics} topics...")
    kmeans = KMeans(n_clusters=n_topics, random_state=42)
    df['dominant_topic'] = kmeans.fit_predict(reduced_embeddings)
    
    # Get the most representative documents for each cluster
    logger.info("Finding most representative documents for each topic...")
    
    # Calculate cluster centers
    cluster_centers = kmeans.cluster_centers_
    
    # Find closest documents to each cluster center
    topic_keywords = {}
    for topic_idx in range(n_topics):
        # Get documents in this cluster
        cluster_docs = df[df['dominant_topic'] == topic_idx]
        
        if len(cluster_docs) > 0:
            # Get the embeddings for these documents
            cluster_embeddings = reduced_embeddings[df['dominant_topic'] == topic_idx]
            
            # Calculate distance to cluster center
            distances = np.linalg.norm(cluster_embeddings - cluster_centers[topic_idx], axis=1)
            
            # Get the indices of the closest documents
            closest_indices = np.argsort(distances)[:5]
            
            # Get the actual document indices in the original dataframe
            doc_indices = cluster_docs.index[closest_indices]
            
            # Extract key terms from these documents
            top_docs_text = ' '.join(df.loc[doc_indices, 'processed_text'])
            
            # Extract most common words
            words = top_docs_text.split()
            word_counts = {}
            for word in words:
                if word not in word_counts:
                    word_counts[word] = 0
                word_counts[word] += 1
            
            # Get top words
            top_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:n_top_words]
            top_words = [word for word, count in top_words]
            
            topic_keywords[topic_idx] = top_words
            logger.info(f"Topic #{topic_idx}: {' '.join(top_words)}")
    
    # Add the topic keywords to each document
    df['topic_keywords'] = df['dominant_topic'].apply(
        lambda x: ', '.join(topic_keywords.get(x, ['unknown']))
    )
    
    return df

# BERTopic-based topic modeling
def perform_bertopic_modeling(df, n_topics=10, n_top_words=10):
    if not HAVE_BERTOPIC:
        logger.warning("BERTopic not available, falling back to embedding-based clustering")
        return perform_embedding_topic_modeling(df, n_topics, n_top_words)
    
    logger.info("Using BERTopic for advanced topic modeling...")
    
    # Create BERTopic model
    topic_model = BERTopic(nr_topics=n_topics, verbose=True)
    
    # Fit the model
    topics, probs = topic_model.fit_transform(df['chunk_text'].tolist())
    
    # Get topic info
    topic_info = topic_model.get_topic_info()
    logger.info("\nTopic Information:")
    logger.info(topic_info.head(n_topics))
    
    # Get topic keywords
    topic_keywords = {}
    for topic_idx in range(-1, min(n_topics, len(topic_model.get_topics()))):
        if topic_idx in topic_model.get_topics():
            words = [word for word, _ in topic_model.get_topic(topic_idx)][:n_top_words]
            topic_keywords[topic_idx] = words
            logger.info(f"Topic #{topic_idx}: {' '.join(words)}")
    
    # Add results to dataframe
    df['dominant_topic'] = topics
    df['topic_probability'] = [max(prob) if len(prob) > 0 else 0 for prob in probs]
    
    # Add the topic keywords to each document
    df['topic_keywords'] = df['dominant_topic'].apply(
        lambda x: ', '.join(topic_keywords.get(x, ['unknown']))
    )
    
    return df

# Alternative: Use CountVectorizer for traditional LDA
def perform_count_based_lda(df, n_topics=10, n_top_words=10):
    logger.info("Preprocessing text data...")
    df['processed_text'] = df['chunk_text'].apply(preprocess_text)
    
    # Filter out empty texts
    df = df[df['processed_text'].str.strip() != '']
    
    logger.info("Vectorizing text data using Count Vectorizer...")
    count_vectorizer = CountVectorizer(
        max_df=0.95,      # Ignore terms that appear in >95% of documents
        min_df=2,         # Ignore terms that appear in <2 documents
        max_features=1000,
        stop_words='english'
    )
    
    count_data = count_vectorizer.fit_transform(df['processed_text'])
    feature_names = count_vectorizer.get_feature_names_out()
    
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
    
    # Print top words for each topic
    topic_keywords = print_top_words(lda, feature_names, n_top_words)
    
    # Get the dominant topic for each document
    logger.info("Assigning dominant topics to each chunk...")
    topic_distributions = lda.transform(count_data)
    df['dominant_topic'] = topic_distributions.argmax(axis=1)
    
    # Add the topic keywords to each document
    df['topic_keywords'] = df['dominant_topic'].apply(
        lambda x: ', '.join(topic_keywords[x])
    )
    
    return df

# LDA Topic Modeling with TF-IDF
def perform_lda_topic_modeling(df, n_topics=10, n_top_words=10):
    logger.info("Preprocessing text data...")
    df['processed_text'] = df['chunk_text'].apply(preprocess_text)
    
    # Filter out empty texts
    df = df[df['processed_text'].str.strip() != '']
    
    logger.info("Vectorizing text data using TF-IDF...")
    tfidf_vectorizer = TfidfVectorizer(
        max_df=0.95,      # Ignore terms that appear in >95% of documents
        min_df=2,         # Ignore terms that appear in <2 documents
        max_features=1000,
        stop_words='english'
    )
    
    tfidf = tfidf_vectorizer.fit_transform(df['processed_text'])
    feature_names = tfidf_vectorizer.get_feature_names_out()
    
    logger.info(f"Fitting LDA model with {n_topics} topics...")
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        max_iter=10,
        learning_method='online',
        learning_offset=50.,
        random_state=0
    )
    
    t0 = time()
    lda.fit(tfidf)
    logger.info(f"LDA model fitted in {time() - t0:.2f}s")
    
    # Print top words for each topic
    topic_keywords = print_top_words(lda, feature_names, n_top_words)
    
    # Get the dominant topic for each document
    logger.info("Assigning dominant topics to each chunk...")
    topic_distributions = lda.transform(tfidf)
    df['dominant_topic'] = topic_distributions.argmax(axis=1)
    
    # Add the topic keywords to each document
    df['topic_keywords'] = df['dominant_topic'].apply(
        lambda x: ', '.join(topic_keywords[x])
    )
    
    return df

# Evaluate topic coherence
def evaluate_topic_quality(df):
    # Group by topic and calculate metrics
    topic_stats = df.groupby('dominant_topic').agg({
        'chunk_text': 'count',
        'processed_text': lambda x: ' '.join(x).split(),
    }).reset_index()
    
    # Rename columns
    topic_stats.columns = ['topic_id', 'document_count', 'word_count']
    
    # Calculate word count
    topic_stats['word_count'] = topic_stats['word_count'].apply(len)
    
    # Calculate average words per document
    topic_stats['avg_words_per_doc'] = topic_stats['word_count'] / topic_stats['document_count']
    
    logger.info("\nTopic Quality Metrics:")
    logger.info(topic_stats)
    
    return topic_stats

def main():
    parser = argparse.ArgumentParser(description='Perform topic modeling on meeting transcript chunks')
    parser.add_argument('--input', type=str, default='data/chunked_transcripts.csv', help='Input CSV file with transcript chunks')
    parser.add_argument('--output', type=str, default='results/embedding_topics_output.csv', help='Output CSV file with topic assignments')
    parser.add_argument('--n_topics', type=int, default=10, help='Number of topics to extract')
    parser.add_argument('--method', type=str, default='lda', 
                        choices=['lda', 'tfidf-lda', 'embeddings', 'bertopic'],
                        help='Topic modeling method')
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
    
    # Perform topic modeling based on selected method
    if args.method == 'lda':
        result_df = perform_count_based_lda(df, n_topics=args.n_topics)
    elif args.method == 'tfidf-lda':
        result_df = perform_lda_topic_modeling(df, n_topics=args.n_topics)
    elif args.method == 'embeddings':
        result_df = perform_embedding_topic_modeling(df, n_topics=args.n_topics)
    elif args.method == 'bertopic':
        result_df = perform_bertopic_modeling(df, n_topics=args.n_topics)
    
    # Evaluate topic quality
    topic_stats = evaluate_topic_quality(result_df)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Save results
    logger.info(f"Saving results to {args.output}...")
    result_df.to_csv(args.output, index=False)
    
    # Save topic statistics
    topic_stats.to_csv(args.output.replace('.csv', '_stats.csv'), index=False)
    
    logger.info("Done!")

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
