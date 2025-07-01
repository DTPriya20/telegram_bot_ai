import os
import fitz  # PyMuPDF
import docx
import faiss
import pickle
import logging
from io import BytesIO
from telegram import Update, Document, InputFile
from telegram.ext import (
    ApplicationBuilder, CommandHandler,
    MessageHandler, filters, ContextTypes
)
from sentence_transformers import SentenceTransformer
from typing import List
from collections import Counter
import re
from sklearn.metrics.pairwise import cosine_similarity
from fpdf import FPDF

# === CONFIGURATION ===
TOKEN = "8125074295:AAHRIlUkN3UlZVao_BYKedAxantp6ts_MUQ"
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
        page_texts = []
        with fitz.open(file_path) as doc:
            for page in doc:
                page_texts.append(page.get_text())
        return page_texts
    elif file_path.endswith(".docx"):
        full_text = [p.text for p in docx.Document(file_path).paragraphs]
        return ["\n".join(full_text)]
    elif file_path.endswith(".md"):
        with open(file_path, "r", encoding="utf-8") as f:
            return [f.read()]
    return []

def chunk_text(text: str, max_words: int = 100) -> List[str]:
    words = text.split()
    return [" ".join(words[i:i+max_words]) for i in range(0, len(words), max_words)]

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

def summarize_passages(passages: List[str], num_points: int = 10) -> str:
    lines = []
    for p in passages:
        lines.extend(re.split(r'[.!?]\s+', p))

    filtered = [line.strip() for line in lines if len(line.strip()) > 40 and not line.strip().isdigit()]
    seen = set()
    unique_lines = [line for line in filtered if not (line in seen or seen.add(line))]

    if not unique_lines:
        return "⚠️ Not enough content to summarize."

    detailed_summary = "\n\n".join([f"🔹 {line}" for line in unique_lines[:num_points]])
    return detailed_summary

def generate_flashcards(chunks: List[str], num_cards: int = 5) -> List[str]:
    flashcards = []
    for chunk in chunks[:num_cards]:
        words = chunk.split()
        question = f"What is: {words[0]}?" if words else "What is this about?"
        answer = chunk[:200] + ("..." if len(chunk) > 200 else "")
        flashcards.append(f"Q: {question}\nA: {answer}")
    all_flashcards.extend(flashcards)
    return flashcards

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

# === HANDLERS ===
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("👋 Hi! Upload a `.pdf`, `.docx`, or `.md` file. Then ask questions based on it!")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = (
        "📘 *Help Menu*\n\n"
        "- Upload a PDF, DOCX, or MD file\n"
        "- Ask questions based on uploaded content\n"
        "- Use 'prepare me for exam' for notes/flashcards\n"
        "- Say 'more flashcards' to get additional ones\n"
        "- Ask: 'give me the answer in 3 points' or '5 points' etc.\n"
        "- Ask: 'summary of page 2' to get that page summary\n"
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
        await update.message.reply_document(
            document=InputFile(pdf_stream, filename="flashcards.pdf"),
            caption="📄 Here are your exported flashcards."
        )
    else:
        await update.message.reply_text("⚠️ No flashcards to export yet.")

async def handle_file(update: Update, context: ContextTypes.DEFAULT_TYPE):
    document: Document = update.message.document
    file_path = os.path.join(DOC_DIR, document.file_name)

    file = await context.bot.get_file(document.file_id)
    await file.download_to_drive(file_path)

    text_pages = extract_text(file_path)
    all_chunks = []
    global passages, page_passages
    for page_text in text_pages:
        chunks = chunk_text(page_text)
        all_chunks.extend(chunks)
        page_passages.append(chunks)

    embs = embed_passages(all_chunks)
    passages.extend(all_chunks)
    index.add(embs)
    save_vector_store()

    preview = summarize_passages(all_chunks[:15])
    await update.message.reply_text(f"✅ Document '{document.file_name}' indexed.\n\n📝 Summary Preview:\n\n{preview}\n\nNow ask your questions!")

async def handle_query(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    query = update.message.text.lower().strip()

    if user_id in user_state:
        state = user_state.pop(user_id)
        if state == "awaiting_exam_choice":
            if "note" in query:
                summary = summarize_passages(passages[:15])
                await update.message.reply_text(f"📝 Concise Notes:\n\n{summary}")
            elif "flash" in query:
                start_idx = user_flashcard_index.get(user_id, 0)
                cards = generate_flashcards(passages[start_idx:], num_cards=5)
                if cards:
                    user_flashcard_index[user_id] = start_idx + 5
                    await update.message.reply_text("📚 Flashcards:\n\n" + "\n\n".join(cards))
                else:
                    await update.message.reply_text("✅ You've reached the end of the flashcards.")
            else:
                user_state[user_id] = "awaiting_exam_choice"
                await update.message.reply_text("❓ Please reply with either 'notes' or 'flashcards'.")
            return

    if query in ["more flashcards", "give me more flashcards", "next flashcards"]:
        start_idx = user_flashcard_index.get(user_id, 0)
        cards = generate_flashcards(passages[start_idx:], num_cards=5)
        if cards:
            user_flashcard_index[user_id] = start_idx + 5
            await update.message.reply_text("📚 More Flashcards:\n\n" + "\n\n".join(cards))
        else:
            await update.message.reply_text("✅ No more flashcards available.")
        return

    if query == "prepare me for exam":
        user_state[user_id] = "awaiting_exam_choice"
        await update.message.reply_text("🧠 Do you want concise notes or flashcards?")
        return

    page_match = re.search(r"page\s+(\d+)", query)
    if page_match:
        page_num = int(page_match.group(1))
        if 1 <= page_num <= len(page_passages):
            selected_chunks = page_passages[page_num - 1]
            point_match = re.search(r"in (\d+) points?", query)
            num_points = int(point_match.group(1)) if point_match else TOP_K
            summary = summarize_passages(selected_chunks, num_points=num_points)
            await update.message.reply_text(f"📄 Summary of page {page_num} in {num_points} points:\n\n{summary}")
        else:
            await update.message.reply_text("⚠️ Page number out of range.")
        return

    if index.ntotal == 0:
        await update.message.reply_text("⚠️ No documents uploaded yet. Please send one first.")
        return

    query_emb = model.encode([query])
    D, I = index.search(query_emb, len(passages))
    matched = [passages[i] for i in I[0] if cosine_similarity([query_emb[0]], [model.encode([passages[i]])[0]])[0][0] > 0.3]

    if not matched:
        await update.message.reply_text("🤖 Sorry, no relevant content found. Try asking differently.")
        return

    point_match = re.search(r"in (\d+) points?", query)
    num_points = int(point_match.group(1)) if point_match else TOP_K

    summary = summarize_passages(matched, num_points=num_points)
    await update.message.reply_text(f"📄 {summary}")

# === MAIN ===
if __name__ == "__main__":
    load_vector_store()

    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(CommandHandler("export", export_flashcards))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_file))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_query))

    print("🤖 Bot is running...")
    app.run_polling()
