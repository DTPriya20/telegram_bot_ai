import os
import fitz  # PyMuPDF
import docx
import faiss
import pickle
import logging
import re
from typing import List, Tuple
from io import BytesIO
from telegram import Update, Document, InputFile
from telegram.ext import (
    ApplicationBuilder, CommandHandler,
    MessageHandler, filters, ContextTypes
)
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from fpdf import FPDF

# === CONFIGURATION ===
TOKEN = "" #for security reasons cannot share the token, you can use your bot token if you want to clone this repo
DOC_DIR = "uploaded_docs"
VEC_STORE = "vector_store.pkl"
FAISS_INDEX = "faiss.index"
EMBED_DIM = 384
TOP_K = 3

# === SETUP ===
logging.basicConfig(level=logging.INFO)
model = SentenceTransformer("all-MiniLM-L6-v2")
if not os.path.exists(DOC_DIR):
    os.makedirs(DOC_DIR)

index = faiss.IndexFlatL2(EMBED_DIM)
passages: List[str] = []
page_passages: List[List[str]] = []
user_state = {}
user_flashcard_index = {}
all_flashcards: List[str] = []

# === HELPERS ===
def extract_text(file_path: str) -> List[str]:
    if file_path.endswith(".pdf"):
        with fitz.open(file_path) as doc:
            return [page.get_text() for page in doc]
    elif file_path.endswith(".docx"):
        full_text = [p.text for p in docx.Document(file_path).paragraphs]
        return ["\n".join(full_text)]
    elif file_path.endswith(".md"):
        with open(file_path, "r", encoding="utf-8") as f:
            return [f.read()]
    return []

def chunk_text(text: str, max_words: int = 250) -> List[str]:
    words = text.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

def embed_passages(chunks: List[str]):
    return model.encode(chunks, show_progress_bar=False)

def save_vector_store():
    faiss.write_index(index, FAISS_INDEX)
    with open(VEC_STORE, "wb") as f:
        pickle.dump(passages, f)

def load_vector_store():
    global passages, index
    if os.path.exists(FAISS_INDEX) and os.path.exists(VEC_STORE):
        index = faiss.read_index(FAISS_INDEX)
        with open(VEC_STORE, "rb") as f:
            passages[:] = pickle.load(f)

def summarize_passages(passages: List[str], query: str = None, full: bool = False, max_points: int = 50) -> List[str]:
    lines = []
    for p in passages:
        lines.extend(re.split(r'(?<=[.!?])\s+', p.strip()))
    filtered = [line.strip() for line in lines if len(line.strip()) > 10 and not line.strip().isdigit()]
    seen = set()
    unique_lines = [line for line in filtered if not (line in seen or seen.add(line))]
    if not unique_lines:
        return ["⚠️ Not enough content to summarize."]
    if query:
        query_emb = model.encode([query])
        line_embs = model.encode(unique_lines)
        sims = cosine_similarity(query_emb, line_embs)[0]
        scored_lines = sorted(zip(unique_lines, sims), key=lambda x: x[1], reverse=True)
        selected = [line for line, _ in scored_lines[:max_points]]
    else:
        selected = unique_lines if full else unique_lines[:max_points]
    messages, current_chunk, current_length = [], [], 0
    TELEGRAM_LIMIT = 3900
    for point in selected:
        bullet = f"🔹 {point.strip()}\n\n"
        if len(bullet) > TELEGRAM_LIMIT:
            continue
        if current_length + len(bullet) > TELEGRAM_LIMIT:
            messages.append("".join(current_chunk).strip())
            current_chunk = [bullet]
            current_length = len(bullet)
        else:
            current_chunk.append(bullet)
            current_length += len(bullet)
    if current_chunk:
        messages.append("".join(current_chunk).strip())
    return messages

def generate_flashcards(chunks: List[str]) -> List[Tuple[str, str]]:
    global all_flashcards
    flashcards = []
    for chunk in chunks:
        sentences = re.split(r'(?<=[.?!])\s+', chunk.strip())
        answer = " ".join(sentences).strip()
        question = None
        for sentence in sentences:
            match = re.match(r"(A|An|The)?\s*(\w+)\s+is\s+", sentence, re.IGNORECASE)
            if match:
                question = f"What is {match.group(2)}?"
                break
        if not question:
            first_sentence = sentences[0].strip()
            first_word = first_sentence.split()[0] if first_sentence else "this"
            question = f"What is {first_word} about?"
        flashcards.append((question, answer))
        all_flashcards.append(f"Q: {question}\nA: {answer}")
    return flashcards

def split_text_by_sentences(text: str, max_chars: int) -> List[str]:
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks, current = [], ""
    for sentence in sentences:
        if len(current) + len(sentence) + 1 > max_chars:
            if current.strip():
                chunks.append(current.strip())
            current = sentence
        else:
            current += " " + sentence
    if current.strip():
        chunks.append(current.strip())
    return chunks

def generate_flashcards_pdf() -> BytesIO:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for card in all_flashcards:
        clean_card = card.encode('latin-1', 'replace').decode('latin-1')
        pdf.multi_cell(0, 10, clean_card + "\n", border=0)
    pdf_stream = BytesIO()
    pdf_bytes = pdf.output(dest='S').encode('latin1')
    pdf_stream.write(pdf_bytes)
    pdf_stream.seek(0)
    return pdf_stream

async def send_summary_chunks(chunks: List[str], update: Update):
    for chunk in chunks:
        await update.message.reply_text(chunk)

async def send_flashcards(cards: List[Tuple[str, str]], update: Update):
    MAX_CHARS = 3900
    flashcard_chunks, current_chunk, current_length = [], [], 0
    for question, answer in cards:
        card_text = f"🔹 *{question.strip()}*\n{answer.strip()}\n\n"
        if len(card_text) > MAX_CHARS:
            split_answer = split_text_by_sentences(answer, MAX_CHARS - len(question) - 10)
            for part in split_answer:
                flashcard_chunks.append(f"🔹 *{question.strip()}*\n{part.strip()}\n\n")
            continue
        if current_length + len(card_text) < MAX_CHARS:
            current_chunk.append(card_text)
            current_length += len(card_text)
        else:
            flashcard_chunks.append("".join(current_chunk).strip())
            current_chunk = [card_text]
            current_length = len(card_text)
    if current_chunk:
        flashcard_chunks.append("".join(current_chunk).strip())
    for chunk in flashcard_chunks:
        await update.message.reply_text(chunk, parse_mode="Markdown")

# === COMMANDS ===
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("👋 Hi! Upload a `.pdf`, `.docx`, or `.md` file. Then ask questions based on it! or type /help to know more!!")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = (
        "📘 *Help Menu*\n\n"
        "- Upload a PDF, DOCX, or MD file\n"
        "- Ask questions based on uploaded content\n"
        "- Use 'prepare me for exam' for notes/flashcards\n"
        "- Say 'next flashcards' to get additional ones\n"
        "- Ask: 'give me the answer in x points' for pointwise summaries\n"
        "- Use /reset to clear all uploaded data\n"
        "- Use /export to download flashcards file"
    )
    await update.message.reply_text(help_text, parse_mode="Markdown")

async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global passages, page_passages, index, all_flashcards
    passages = []
    page_passages = []
    all_flashcards = []
    index = faiss.IndexFlatL2(EMBED_DIM)
    await update.message.reply_text("🧹 Reset complete. Upload a document to start again.")

async def export_flashcards(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if all_flashcards:
        pdf_stream = generate_flashcards_pdf()
        await update.message.reply_document(InputFile(pdf_stream, filename="flashcards.pdf"))
    else:
        await update.message.reply_text("⚠️ No flashcards to export yet.")

# === FILE & QUERY HANDLERS ===
async def handle_file(update: Update, context: ContextTypes.DEFAULT_TYPE):
    document: Document = update.message.document
    file_path = os.path.join(DOC_DIR, document.file_name)

    #  Await the coroutine and call download_to_drive properly
    telegram_file = await context.bot.get_file(document.file_id)
    await telegram_file.download_to_drive(file_path)

    text_pages = extract_text(file_path)
    all_chunks = []
    global passages, page_passages
    passages.clear()
    page_passages.clear()
    for page in text_pages:
        chunks = chunk_text(page)
        passages.extend(chunks)
        page_passages.append(chunks)

    embs = embed_passages(passages)
    index.add(embs)
    save_vector_store()
    preview = summarize_passages(passages, full=True)
    await update.message.reply_text(f"✅ Document '{document.file_name}' indexed.\n\n📝 Summary Preview:")
    await send_summary_chunks(preview, update)


async def handle_query(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.message.text.lower().strip()
    user_id = update.message.from_user.id

    if query == "prepare me for exam":
        user_state[user_id] = "exam"
        await update.message.reply_text("📚 Do you want 'notes' or 'flashcards'?")
        return
    elif query == "more flashcards":
        start = user_flashcard_index.get(user_id, 0)
        cards = generate_flashcards(passages[start:start + 10])
        user_flashcard_index[user_id] = start + 10
        await send_flashcards(cards, update)
        return
    elif user_state.get(user_id) == "exam":
        user_state.pop(user_id)
        if "note" in query:
            await send_summary_chunks(summarize_passages(passages, full=True), update)
        elif "flash" in query:
            cards = generate_flashcards(passages[:10])
            user_flashcard_index[user_id] = 10
            await send_flashcards(cards, update)
        else:
            await update.message.reply_text("❓ Reply with 'notes' or 'flashcards'")
        return

    page_match = re.search(r"page\s+(\d+)", query)
    if page_match:
        num = int(page_match.group(1))
        if 1 <= num <= len(page_passages):
            await send_summary_chunks(summarize_passages(page_passages[num - 1], query=query), update)
        else:
            await update.message.reply_text("⚠️ Page number out of range.")
        return

    points_match = re.search(r"in (\d+) points?", query)
    num_points = int(points_match.group(1)) if points_match else 30

    if index.ntotal == 0:
        await update.message.reply_text("⚠️ No documents uploaded yet.")
        return

    query_emb = model.encode([query])
    D, I = index.search(query_emb, min(len(passages), TOP_K * 3))
    matched = [passages[i] for i in I[0] if 0 <= i < len(passages)]
    if not matched:
        matched = passages[:TOP_K * 3]
    summary = summarize_passages(matched, query=query, max_points=num_points)
    await send_summary_chunks(summary, update)

# === MAIN ===
if __name__ == "__main__":
    load_vector_store()
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(CommandHandler("export", export_flashcards))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_file))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_query))
    print("🤖 Bot is running...")
    app.run_polling()
