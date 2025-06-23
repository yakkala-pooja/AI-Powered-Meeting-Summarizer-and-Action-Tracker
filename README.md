# AI-Powered Meeting Summarizer and Action Tracker

<div align="center">

![Meeting Summarizer Logo](frontend/public/logo192.png)

**Transform Meeting Transcripts into Actionable Intelligence**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![React 18](https://img.shields.io/badge/react-18-61DAFB.svg)](https://reactjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95+-009688.svg)](https://fastapi.tiangolo.com/)

</div>

## 🔍 Project Overview

In today's fast-paced work environment, meetings consume a significant portion of our professional lives, yet their outputs often vanish into digital oblivion. The AI-Powered Meeting Summarizer and Action Tracker bridges this gap by transforming raw meeting transcripts into structured, actionable intelligence.

This application leverages the power of Large Language Models (LLMs) to analyze meeting transcripts and automatically extract three critical components:

- **📋 Meeting Summaries**: Concise overviews that capture the essence of discussions
- **🤝 Decisions**: Clear record of agreements, votes, and conclusions reached
- **✅ Action Items**: Explicit tasks with ownership and deadlines

What makes this tool unique is its flexibility in AI processing. You can run it completely locally using Ollama for privacy-sensitive content, leverage cloud-based models like GPT for enhanced accuracy, or fall back to rule-based extraction when AI services are unavailable.

<div align="center">
  <img src="results/meeting_dataset_analysis.png" alt="Meeting Analysis Example" width="700px">
</div>

## ✨ Features

- **🧠 Multiple AI Options**: 
  - Local processing with Ollama models (Mistral, Llama, Phi, etc.)
  - Cloud processing with OpenAI API (GPT-3.5, GPT-4)
  - Rule-based fallback extraction
- **💻 Modern UI**: Responsive React frontend with dark mode support
- **⚡ High Performance**: FastAPI backend with caching for rapid analysis
- **🔄 Flexible Processing**: Process entire documents or chunk large transcripts
- **📊 Advanced Analysis**: Topic modeling and sentiment analysis capabilities
- **📧 Sharing**: Email integration to distribute meeting insights

## 🚀 Getting Started

### Prerequisites

- Python 3.8+ 
- Node.js 14+ and npm
- [Ollama](https://ollama.com/download) for local LLM processing (optional)
- OpenAI API key (optional)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/AI-Powered-Meeting-Summarizer-and-Action-Tracker.git
   cd AI-Powered-Meeting-Summarizer-and-Action-Tracker
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up AI backend** (choose one or both):
   
   **Option 1: Local LLMs with Ollama**
   ```bash
   # For macOS/Linux
   curl -fsSL https://ollama.com/install.sh | sh
   
   # For Windows
   # Download from https://ollama.com/download and follow installation instructions
   
   # Pull required models
   ollama pull llama3.2  # Best quality results (8GB RAM)
   ollama pull mistral   # Good balance (4.8GB RAM)
   ollama pull phi       # Lightweight option (3GB RAM)
   ```
   
   **Option 2: OpenAI API**
   
   Create a `.env` file in the project root with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

4. **Install frontend dependencies**:
   ```bash
   cd frontend
   npm install
   cd ..
   ```

## 💻 Usage

### Starting the Application

1. **Start the backend API**:
   ```bash
   python api.py
   ```
   The API will be available at http://localhost:8000

2. **Start the frontend** (in a new terminal):
   ```bash
   cd frontend
   npm start
   ```
   The application will be available at http://localhost:3000

### Using the Application

1. **Upload a transcript**: Click "Select File" to upload a meeting transcript (.txt or .md)
2. **Select model**: Choose the LLM model to use for analysis
3. **Analyze**: Click "Analyze Transcript" to process the meeting
4. **View results**: See the extracted summary, decisions, and action items
5. **Share**: Email the results to meeting participants if needed

### Command Line Usage

Process meeting transcripts directly from the command line:

```bash
# Using Ollama with default settings (single document mode)
python scripts/extract_meeting_info.py --input data/chunked_transcripts.csv --output results/meeting_extractions.csv --model llama3.2

# Enable chunking for very large documents
python scripts/extract_meeting_info.py --input data/chunked_transcripts.csv --output results/meeting_extractions.csv --model mistral --use_chunking
```

## 🔧 Configuration

### Model Selection Guide

#### Ollama Models (Local)

| Model | RAM Required | Speed | Quality | Best For |
|-------|-------------|-------|---------|----------|
| llama3.2 | ~8GB | Slower | Excellent | Detailed analysis of complex meetings |
| mistral | ~4.8GB | Medium | Very Good | General purpose meeting analysis |
| phi | ~3GB | Fast | Good | Quick analysis of shorter meetings |
| gemma | ~4GB | Medium | Good | Balanced option for most meetings |

#### OpenAI Models (Cloud)

| Model | Cost | Speed | Quality | Best For |
|-------|------|-------|---------|----------|
| gpt-3.5-turbo | $ | Fast | Very High | Regular meeting analysis |
| gpt-4 | $$$ | Medium | Exceptional | Critical meetings with complex content |

## 📊 Analysis Capabilities

### Meeting Summary Extraction

The system identifies the most important points discussed in the meeting, focusing on substantive content rather than procedural remarks. It creates a coherent paragraph that captures the essence of the discussion.

### Decision Extraction

The system recognizes various types of decisions:
- Formal votes and approvals
- Consensus agreements
- Policy adoptions
- Resource allocations

### Action Item Extraction

The system identifies tasks that need to be completed, including:
- Assigned responsibilities (who is accountable)
- Deadlines and timeframes
- Required approvals
- Scheduled follow-ups

## 🧩 Interesting Technical Challenges

### 1. Balancing Precision and Recall in Action Item Detection

One of the most fascinating challenges in this project was finding the right balance between precision and recall when identifying action items. Too strict, and we'd miss subtle task assignments; too loose, and regular statements would be misclassified as tasks.

Our solution involved a multi-layered approach:
- Pattern matching for explicit markers ("to-do", "action item", "next steps")
- Named entity recognition to identify person-task relationships
- Contextual analysis of imperative verbs and future tense statements
- Deadline and temporal expression detection

### 2. Handling Malformed LLM Responses

Large Language Models occasionally produce malformed JSON or nested structures that break standard parsers. We implemented a robust response handling system that:
- Detects and repairs common JSON formatting issues
- Handles nested JSON objects by flattening them
- Provides multiple fallback parsing strategies
- Gracefully degrades to rule-based extraction when necessary

### 3. Optimizing for Different Document Sizes

Meeting transcripts vary dramatically in size - from brief stand-ups to day-long workshops. Our processing pipeline needed to handle both extremes efficiently:
- For smaller documents: Single-pass processing for coherent analysis
- For larger documents: Intelligent chunking with context preservation
- Memory-efficient processing to avoid OOM errors
- Caching system to avoid reprocessing similar content

### 4. Cross-Model Consistency

Ensuring consistent output quality across different LLM models (from small local models to powerful cloud APIs) required careful prompt engineering:
- Model-specific prompt templates optimized for each architecture
- Explicit formatting instructions to guide output structure
- Post-processing normalization to standardize outputs
- Quality thresholds to trigger fallback mechanisms when needed

## 🔮 Future Directions

- **Meeting Comparison**: Analyze trends across multiple meetings over time
- **Voice Recognition**: Direct audio-to-summary pipeline
- **Participant Analysis**: Track speaker contributions and engagement
- **Integration with Meeting Platforms**: Direct plugins for Zoom, Teams, etc.
- **Custom Domain Adaptation**: Fine-tuning for specific industries or meeting types

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- The Ollama team for making local LLMs accessible
- FastAPI and React communities for excellent frameworks
- Contributors who have helped improve this tool