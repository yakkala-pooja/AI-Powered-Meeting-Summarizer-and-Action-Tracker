import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
import re
import os

# Load the cleaned data
print("Loading cleaned meeting dataset...")
df = pd.read_csv("data/meeting_summaries_clean.csv")

# Display basic information
print(f"\nDataset shape: {df.shape}")
print("\nColumn information:")
print(df.info())

# Calculate statistics on text length
print("\nCalculating text length statistics...")
df['transcript_length'] = df['meeting_transcript'].apply(len)
df['summary_length'] = df['summary'].apply(len)
df['compression_ratio'] = df['summary_length'] / df['transcript_length']

print("\nTranscript length statistics:")
print(df['transcript_length'].describe())

print("\nSummary length statistics:")
print(df['summary_length'].describe())

print("\nCompression ratio statistics (summary length / transcript length):")
print(df['compression_ratio'].describe())

# Find examples with shortest and longest transcripts
shortest_idx = df['transcript_length'].idxmin()
longest_idx = df['transcript_length'].idxmax()

print(f"\nShortest transcript (length: {df.loc[shortest_idx, 'transcript_length']}):")
print(df.loc[shortest_idx, 'meeting_transcript'][:500] + "..." if len(df.loc[shortest_idx, 'meeting_transcript']) > 500 else df.loc[shortest_idx, 'meeting_transcript'])
print(f"\nSummary of shortest transcript (length: {df.loc[shortest_idx, 'summary_length']}):")
print(df.loc[shortest_idx, 'summary'][:500] + "..." if len(df.loc[shortest_idx, 'summary']) > 500 else df.loc[shortest_idx, 'summary'])

print(f"\nLongest transcript (length: {df.loc[longest_idx, 'transcript_length']}, showing first 500 chars):")
print(df.loc[longest_idx, 'meeting_transcript'][:500] + "...")
print(f"\nSummary of longest transcript (length: {df.loc[longest_idx, 'summary_length']}, showing first 500 chars):")
print(df.loc[longest_idx, 'summary'][:500] + "..." if len(df.loc[longest_idx, 'summary']) > 500 else df.loc[longest_idx, 'summary'])

# Create visualizations
print("\nCreating visualizations...")

plt.figure(figsize=(12, 8))

# Histogram of transcript lengths
plt.subplot(2, 2, 1)
sns.histplot(df['transcript_length'], kde=True)
plt.title('Distribution of Transcript Lengths')
plt.xlabel('Length (characters)')
plt.ylabel('Count')

# Histogram of summary lengths
plt.subplot(2, 2, 2)
sns.histplot(df['summary_length'], kde=True)
plt.title('Distribution of Summary Lengths')
plt.xlabel('Length (characters)')
plt.ylabel('Count')

# Scatter plot of transcript length vs summary length
plt.subplot(2, 2, 3)
sns.scatterplot(x='transcript_length', y='summary_length', data=df, alpha=0.5)
plt.title('Transcript Length vs Summary Length')
plt.xlabel('Transcript Length (characters)')
plt.ylabel('Summary Length (characters)')

# Histogram of compression ratios
plt.subplot(2, 2, 4)
sns.histplot(df['compression_ratio'], kde=True)
plt.title('Distribution of Compression Ratios')
plt.xlabel('Compression Ratio (summary/transcript)')
plt.ylabel('Count')

plt.tight_layout()

# Ensure results directory exists
os.makedirs('results', exist_ok=True)
plt.savefig('results/meeting_dataset_analysis.png')
print("Saved visualizations to 'results/meeting_dataset_analysis.png'")

# Extract common phrases in summaries
def extract_common_phrases(texts, min_length=3, max_length=5, top_n=20):
    all_phrases = []
    for text in texts:
        words = re.findall(r'\b\w+\b', text.lower())
        for i in range(len(words) - min_length + 1):
            for j in range(min_length, min(max_length + 1, len(words) - i + 1)):
                phrase = ' '.join(words[i:i+j])
                all_phrases.append(phrase)
    
    return Counter(all_phrases).most_common(top_n)

print("\nExtracting common phrases from summaries...")
common_phrases = extract_common_phrases(df['summary'].tolist(), min_length=3, max_length=5, top_n=20)
print("\nTop 20 common phrases in summaries:")
for phrase, count in common_phrases:
    print(f"'{phrase}': {count} occurrences")

print("\nAnalysis complete!") 