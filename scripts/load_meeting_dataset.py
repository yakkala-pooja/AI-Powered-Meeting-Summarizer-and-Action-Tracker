import pandas as pd
from datasets import load_dataset
import os

# Load the MeetingBank-QA-Summary dataset from Hugging Face
print("Loading MeetingBank-QA-Summary dataset...")
dataset = load_dataset("microsoft/MeetingBank-QA-Summary")

# Print dataset structure
print("\nDataset structure:")
print(dataset)

# Convert to pandas DataFrame
df = pd.DataFrame(dataset["test"])

# Preview raw data
print("\nRaw data preview (first 2 examples):")
for i in range(min(2, len(dataset["test"]))):
    print(f"\nExample {i+1}:")
    print(f"Transcript (first 300 chars): {dataset['test'][i]['prompt'][:300]}...")
    print(f"Summary (first 300 chars): {dataset['test'][i]['summary'][:300]}...")

# Create a new DataFrame with just the required columns
print("\nCreating DataFrame with required columns...")
meeting_df = pd.DataFrame({
    "meeting_transcript": df["prompt"],
    "summary": df["summary"]
})

# Remove any empty or malformed entries
print("\nCleaning data...")
meeting_df_clean = meeting_df.dropna(subset=["meeting_transcript", "summary"])
meeting_df_clean = meeting_df_clean[meeting_df_clean["meeting_transcript"].str.strip() != ""]
meeting_df_clean = meeting_df_clean[meeting_df_clean["summary"].str.strip() != ""]

# Print information about the cleaned DataFrame
print(f"\nCleaned DataFrame shape: {meeting_df_clean.shape}")
print("\nCleaned DataFrame preview:")
print(meeting_df_clean.head(3))

# Ensure data directory exists
os.makedirs('data', exist_ok=True)

# Save the cleaned DataFrame to CSV
meeting_df_clean.to_csv("data/meeting_summaries_clean.csv", index=False)
print("\nSaved cleaned data to 'data/meeting_summaries_clean.csv'") 