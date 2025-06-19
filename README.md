# AI-Powered Meeting Summarizer and Action Tracker

This project implements topic modeling techniques for analyzing meeting transcripts, identifying key topics, and evaluating topic model quality.

## Project Structure

```
AI-Powered-Meeting-Summarizer-and-Action-Tracker/
├── data/                   # Data files
│   ├── chunked_transcripts.csv      # Chunked meeting transcripts
│   └── meeting_summaries_clean.csv  # Original meeting data with transcripts and summaries
│
├── models/                 # Saved models (currently empty)
│
├── results/                # Output files and visualizations
│   ├── meeting_dataset_analysis.png # Visualizations of dataset statistics
│   ├── topic_evaluation_metrics.png # Topic model evaluation metrics plot
│   ├── topic_evaluation_results.csv # Evaluation metrics for different topic counts
│   ├── topic_results.csv            # Topic modeling results
│   └── topic_results_stats.csv      # Topic quality statistics
│
├── scripts/                # Python scripts
│   ├── analyze_meeting_dataset.py   # Analyze and visualize dataset statistics
│   ├── chunk_transcripts.py         # Split transcripts into manageable chunks
│   ├── embedding_topic_modeling.py  # Advanced topic modeling with embeddings
│   ├── evaluate_topic_models.py     # Evaluate different topic model configurations
│   ├── load_meeting_dataset.py      # Load and preprocess meeting dataset
│   └── topic_modeling.py            # Traditional LDA topic modeling
│
└── requirements.txt        # Python dependencies
```

## Getting Started

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Load and preprocess the dataset:
   ```
   python scripts/load_meeting_dataset.py
   ```

3. Chunk the transcripts:
   ```
   python scripts/chunk_transcripts.py
   ```

4. Analyze the dataset:
   ```
   python scripts/analyze_meeting_dataset.py
   ```

5. Evaluate topic models:
   ```
   python scripts/evaluate_topic_models.py
   ```

## Topic Modeling Approaches

This project provides two different implementations of topic modeling to accommodate different needs:

### Traditional Topic Modeling (Optional)

File: `scripts/topic_modeling.py`

```
python scripts/topic_modeling.py --n_topics 10
```

**When to use:**
- For quick exploration of large document collections
- When computational resources are limited
- For simpler, well-structured documents
- When processing speed is a priority

### Embedding-Based Topic Modeling (Recommended)

File: `scripts/embedding_topic_modeling.py`

```
python scripts/embedding_topic_modeling.py --method embeddings --n_topics 10
```

**When to use:**
- For higher quality, more coherent topics
- For conversational text like meeting transcripts
- When semantic relationships between words are important
- When you have access to sufficient computational resources

For meeting transcripts specifically, embedding-based methods are recommended as they better capture the contextual nature of conversations.

## Topic Modeling Methods

- **Traditional LDA**: Uses CountVectorizer or TF-IDF with Latent Dirichlet Allocation
- **Embedding-based**: Uses sentence embeddings for more semantically meaningful topics
- **BERTopic**: Uses transformer models for state-of-the-art topic modeling

## Evaluation Metrics

- **Coherence**: Measures semantic similarity within topics (higher is better)
- **Diversity**: Measures how distinct topics are from each other (higher is better)
- **Perplexity**: Measures how well the model fits the data (lower is better)