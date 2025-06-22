# Meeting Transcript Analyzer Frontend

This is the React frontend for the Meeting Transcript Analyzer application. It allows users to upload meeting transcripts, analyze them using the FastAPI backend, and export the results in various formats.

## Features

- Upload meeting transcripts (.txt or .md files)
- Extract summaries, decisions, and action items using LLMs
- Choose between different LLM models (Mistral, Llama 3.2, etc.)
- Export results as Markdown, HTML, or PDF
- Responsive Material UI design

## Setup

1. Install dependencies:
```bash
cd frontend
npm install
```

2. Start the development server:
```bash
npm start
```

The frontend will be available at http://localhost:3000

## Requirements

- Node.js 14+ and npm
- The FastAPI backend running at http://localhost:8000

## Available Scripts

- `npm start`: Runs the app in development mode
- `npm build`: Builds the app for production
- `npm test`: Runs tests
- `npm eject`: Ejects from create-react-app

## Dependencies

- React
- Material UI
- jsPDF (for PDF export)
- html2canvas (for PDF export)
- file-saver (for file downloads)

## Project Structure

- `src/components/TranscriptAnalyzer.jsx`: Main component for transcript analysis
- `src/App.jsx`: Root component with theme configuration
- `src/index.js`: Entry point

## Usage

1. Start the FastAPI backend (`python api.py`)
2. Start the React frontend (`npm start`)
3. Upload a transcript file (.txt or .md)
4. Select a model and click "Analyze Transcript"
5. View the results and export in your preferred format 