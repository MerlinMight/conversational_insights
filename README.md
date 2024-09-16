# Conversational Insights Platform

This project is a **Conversational Insights Platform** that processes audio/video files to provide comprehensive insights through transcription, topic extraction, sentiment analysis, and more. It integrates a Large Language Model (LLM) for enhanced context-aware analysis, providing summaries, answering questions, and generating reports. The platform is designed for use in retail customer-agent conversations.

## Features

- **Data Ingestion**: Handles audio/video file uploads.
- **Transcription**: Converts speech into text.
- **Topic Extraction**: Identifies key topics discussed in the conversation.
- **Sentiment and Emotion Analysis**: Detects sentiment and emotion distribution in conversations.
- **Metadata Extraction**: Extracts speaker information and conversation metadata.
- **Insight Generation**: Summarizes insights such as customer satisfaction and common complaints.
- **LLM Integration**: Summarizes conversations, answers questions, and performs fact extraction.
- **Web Integration**: Displays outputs through a web interface.

## Directory Structure

```bash
DataScientistAssignment/
├── app.py
├── module/
│   ├── __init__.py
│   ├── data_ingestion.py
├── templates/
│   ├── index.html
│   ├── results.html
├── static/
│   ├── style.css
├── topic_extraction.py
├── metadata_extraction.py
├── transcription.py
├── sentimental_analysis.py
├── insight_generation.py
└── requirements.txt

## Installation

### 1. Clone the Repository

To get started, clone this repository to your local machine:

```bash
git clone <repository_url>
cd DataScientistAssignment

### 2. Set up Virtual Environment (Optional but Recommended)
Create a virtual environment to manage dependencies:

For Linux/Mac:
```bash
python3 -m venv venv
source venv/bin/activate

For Windows:
```bash
python -m venv venv
venv\Scripts\activate


