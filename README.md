# AI Career Coach

AI Career Coach is a Flask-based web app that analyzes a PDF resume and answers follow-up questions using local Ollama models instead of the OpenAI API.

The app:
- uploads a resume PDF
- extracts readable text from the PDF
- generates a resume summary
- lets the user ask questions about the uploaded resume

## Tech Stack

- Python
- Flask
- Ollama `0.6.4`
- LangChain
- FAISS
- PyPDF2
- Tailwind CSS

## Features

- Resume upload with PDF validation
- Resume text extraction
- AI-generated resume summary
- Question-answering based on the uploaded resume
- Local `.env` configuration
- Local inference through Ollama

## Project Structure

```text
Ai_Career_Coach/
|-- app.py
|-- requirements.txt
|-- .env.example
|-- uploads/
|-- vector_index/
|-- latest_resume.txt
`-- template/
    |-- index.html
    |-- ask.html
    |-- result.html
    `-- qa_result.html
```

## Prerequisites

Make sure these are installed on your machine:

- Python 3.10+
- Ollama `0.6.4`

## Ollama Models

This project uses:

- `llama3.2` for resume summary and question answering
- `nomic-embed-text` for embeddings

Pull them before running the app:

```powershell
ollama pull llama3.2
ollama pull nomic-embed-text
```

To verify the models are available:

```powershell
ollama list
```

## Installation

Clone the repository and move into the project folder:

```powershell
git clone <your-repo-url>
cd Ai_Career_Coach
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

If `pip` does not work in your environment, use:

```powershell
py -m pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in the project root. A sample is already provided in `.env.example`.

Example:

```env
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=llama3.2
OLLAMA_EMBED_MODEL=nomic-embed-text
```

## How to Run

Start Ollama first, then run the Flask app:

```powershell
python app.py
```

If `python` does not work:

```powershell
py app.py
```

Then open:

```text
http://127.0.0.1:5000
```

## How to Use

1. Open the home page.
2. Upload a PDF resume.
3. Wait for the app to generate the resume summary.
4. Click the ask-question page.
5. Ask questions related to the uploaded resume.

## How It Works

1. The uploaded PDF is saved in `uploads/`.
2. Resume text is extracted using `PyPDF2`.
3. The extracted text is stored in `latest_resume.txt`.
4. Embeddings are generated using Ollama.
5. A FAISS index is created and saved in `vector_index/`.
6. Ollama generates:
   - a resume summary
   - answers to resume-related questions

## Common Issues

### 1. Model not found

Error example:

```text
model "nomic-embed-text" not found
```

Fix:

```powershell
ollama pull nomic-embed-text
ollama pull llama3.2
```

### 2. Ollama not running

If the app cannot connect to Ollama, make sure Ollama is running locally on:

```text
http://127.0.0.1:11434
```

### 3. Ask feature not working

The ask feature depends on an uploaded resume.

Fix:
- upload a resume first
- make sure the PDF contains selectable text
- re-upload the file if needed

### 4. PDF uploads but no result appears

Some resumes are image-based PDFs and do not contain extractable text.

Fix:
- use a text-based PDF
- or run OCR on the PDF before uploading it

## Free Deployment Note

This project is best suited for local use because it depends on a local Ollama server.

Most free hosting platforms do not support running Ollama models reliably. If you want deployment:

- host Flask separately
- host Ollama on your own machine or a VPS

## Future Improvements

- add OCR support for scanned resumes
- add file upload progress and better error messages
- support multiple resume sessions
- deploy with a remote model endpoint

## License

Use this project as needed for learning and personal projects.
