import os
from pathlib import Path

import PyPDF2
from dotenv import load_dotenv
from flask import Flask, redirect, render_template, request, url_for
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from werkzeug.utils import secure_filename


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

app = Flask(__name__, template_folder="template")

UPLOAD_FOLDER = BASE_DIR / "uploads"
VECTOR_INDEX_FOLDER = BASE_DIR / "vector_index"
ALLOWED_EXTENSIONS = {"pdf"}

app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)

UPLOAD_FOLDER.mkdir(exist_ok=True)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", OLLAMA_MODEL)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
embeddings = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL, base_url=OLLAMA_BASE_URL)
llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)

resume_summary_template = """
Role: You are an AI Career Coach.

Task: Given the candidate's resume, provide a comprehensive summary that includes:

- Career Objective
- Skills and Expertise
- Professional Experience
- Educational Background
- Notable Achievements

Instructions:
Provide a concise, well-structured summary that highlights strengths and career trajectory.

Resume:
{resume}
"""

resume_prompt = PromptTemplate(
    input_variables=["resume"],
    template=resume_summary_template,
)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_text_from_pdf(pdf_path: Path) -> str:
    with pdf_path.open("rb") as file:
        reader = PyPDF2.PdfReader(file)
        pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages).strip()


def build_vector_store(resume_text: str) -> None:
    chunks = text_splitter.split_text(resume_text)
    vectorstore = FAISS.from_texts(chunks, embeddings)
    vectorstore.save_local(str(VECTOR_INDEX_FOLDER))


def summarize_resume(resume_text: str) -> str:
    prompt = resume_prompt.format(resume=resume_text)
    return llm.invoke(prompt).content


def format_ollama_error(error: Exception) -> str:
    message = str(error)

    if "model" in message and "not found" in message:
        return (
            "Ollama model not found. Run `ollama pull llama3.2` and "
            "`ollama pull nomic-embed-text`, then restart the app."
        )

    if "Failed to connect" in message or "Connection refused" in message:
        return (
            "Could not connect to Ollama. Start Ollama first and make sure it is "
            "available on http://127.0.0.1:11434."
        )

    return f"Ollama request failed: {message}"


def perform_qa(query: str) -> str:
    if not VECTOR_INDEX_FOLDER.exists():
        return "Upload a resume first so I can answer questions about it."

    vectorstore = FAISS.load_local(
        str(VECTOR_INDEX_FOLDER),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    documents = vectorstore.similarity_search(query, k=4)

    if not documents:
        return "I could not find relevant resume content for that question."

    context = "\n\n".join(doc.page_content for doc in documents)
    qa_prompt = f"""
You are an AI career coach answering questions about a candidate resume.

Use only the resume context below. If the answer is not present, say that clearly.

Question:
{query}

Resume Context:
{context}
"""
    response = llm.invoke(qa_prompt)
    return response.content if hasattr(response, "content") else str(response)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return redirect(url_for("index"))

    file = request.files["file"]
    if file.filename == "":
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        return render_template(
            "result.html",
            resume_analysis="Only PDF files are supported.",
        )

    filename = secure_filename(file.filename)
    file_path = UPLOAD_FOLDER / filename
    file.save(file_path)

    resume_text = extract_text_from_pdf(file_path)
    if not resume_text:
        return render_template(
            "result.html",
            resume_analysis="No readable text was found in the uploaded PDF.",
        )

    try:
        build_vector_store(resume_text)
        resume_analysis = summarize_resume(resume_text)
    except Exception as error:
        return render_template(
            "result.html",
            resume_analysis=format_ollama_error(error),
        )

    return render_template("result.html", resume_analysis=resume_analysis)


@app.route("/ask", methods=["GET", "POST"])
def ask_query():
    if request.method == "POST":
        query = request.form["query"]
        try:
            result = perform_qa(query)
        except Exception as error:
            result = format_ollama_error(error)
        return render_template("qa_result.html", query=query, result=result)
    return render_template("ask.html")


if __name__ == "__main__":
    app.run(debug=True)
