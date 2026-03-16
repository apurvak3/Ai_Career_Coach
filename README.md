# Ai_Career_Coach

This project now uses a local Ollama server instead of the OpenAI API.

## Setup

1. Install Ollama `0.6.4`.
2. Pull the models defined in `.env`:
   `ollama pull llama3.2`
   `ollama pull nomic-embed-text`
3. Install Python dependencies:
   `pip install -r requirements.txt`
4. Start Ollama, then run:
   `python app.py`
