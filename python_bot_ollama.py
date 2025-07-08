#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# === IMPORTS ===
import os
import json              # NEW
import fitz              # PyMuPDF
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
import ollama            # NEW  ← pip install ollama

# === CONFIGURATION ===
TOKEN       = "" #removed for security reasons
DOC_DIR     = "uploaded_docs"
VEC_STORE   = "vector_store.pkl"
FAISS_INDEX = "faiss.index"
EMBED_DIM   = 384
TOP_K       = 3

# === SETUP ===
logging.basicConfig(level=logging.INFO)
model = SentenceTransformer("all-MiniLM-L6-v2")
if not os.path.exists(DOC_DIR):
    os.makedirs(DOC_DIR)

index: faiss.Index = faiss.IndexFlatL2(EMBED_DIM)
passages: List[str] = []
page_passages: List[List[str]] = []
user_state = {}
user_flashcard_index = {}
all_flashcards: List[str] = []

# ------------------------------------------------------------------ #
#                           HELPER FUNCTIONS                         #
# ------------------------------------------------------------------ #
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

def summarize_passages(passages: List[str], query: str = None,
                       full: bool = False, max_points: int = 50) -> List[str]:
    lines = []
    for p in passages:
        lines.extend(re.split(r'(?<=[.!?])\s+', p.strip()))
    filtered = [ln.strip() for ln in lines
                if len(ln.strip()) > 10 and not ln.strip().isdigit()]
    seen = set()
    unique_lines = [ln for ln in filtered if not (ln in seen or seen.add(ln))]
    if not unique_lines:
        return ["⚠️ Not enough content to summarize."]
    if query:
        query_emb  = model.encode([query])
        line_embs  = model.encode(unique_lines)
        sims       = cosine_similarity(query_emb, line_embs)[0]
        scored     = sorted(zip(unique_lines, sims),
                            key=lambda x: x[1], reverse=True)
        selected   = [line for line, _ in scored[:max_points]]
    else:
        selected = unique_lines if full else unique_lines[:max_points]

    TELEGRAM_LIMIT = 3900
    messages, current_chunk, current_len = [], [], 0
    for point in selected:
        bullet = f"🔹 {point.strip()}\n\n"
        if len(bullet) > TELEGRAM_LIMIT:
            continue
        if current_len + len(bullet) > TELEGRAM_LIMIT:
            messages.append("".join(current_chunk).strip())
            current_chunk = [bullet]
            current_len   = len(bullet)
        else:
            current_chunk.append(bullet)
            current_len += len(bullet)
    if current_chunk:
        messages.append("".join(current_chunk).strip())
    return messages

# ---------------------  LLM   ------------------------------------- #
def call_local_llm(prompt: str,
                   system: str = "You are a helpful study assistant.") -> str:
    """Send a prompt to the locally‑running Ollama model and return reply."""
    try:
        msgs = [
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt}
        ]
        rsp = ollama.chat(model="mistral", messages=msgs)
        return rsp["message"]["content"].strip()
    except Exception as e:
        logging.error(f"LLM error → {e}")
        return ""

# ------------------  Flashcards (LLM‑first) ----------------------- #
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
# --------------------------------------------------------------- #

def split_text_by_sentences(text: str, max_chars: int) -> List[str]:
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks, current = [], ""
    for sent in sentences:
        if len(current) + len(sent) + 1 > max_chars:
            if current.strip():
                chunks.append(current.strip())
            current = sent
        else:
            current += " " + sent
    if current.strip():
        chunks.append(current.strip())
    return chunks

def generate_flashcards_pdf() -> BytesIO:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for card in all_flashcards:
        clean = card.encode("latin-1", "replace").decode("latin-1")
        pdf.multi_cell(0, 10, clean + "\n", border=0)
    stream = BytesIO()
    pdf_bytes = pdf.output(dest="S").encode("latin1")
    stream.write(pdf_bytes)
    stream.seek(0)
    return stream

async def send_summary_chunks(chunks: List[str], update: Update):
    for ch in chunks:
        await update.message.reply_text(ch)

async def send_flashcards(cards: List[Tuple[str, str]], update: Update):
    MAX_CHARS = 3900
    fc_chunks, current_chunk, current_len = [], [], 0
    for q, a in cards:
        text = f"🔹 *{q.strip()}*\n{a.strip()}\n\n"
        if len(text) > MAX_CHARS:
            parts = split_text_by_sentences(a, MAX_CHARS - len(q) - 10)
            for part in parts:
                fc_chunks.append(f"🔹 *{q.strip()}*\n{part.strip()}\n\n")
            continue
        if current_len + len(text) < MAX_CHARS:
            current_chunk.append(text)
            current_len += len(text)
        else:
            fc_chunks.append("".join(current_chunk).strip())
            current_chunk = [text]
            current_len  = len(text)
    if current_chunk:
        fc_chunks.append("".join(current_chunk).strip())
    for ch in fc_chunks:
        await update.message.reply_text(ch, parse_mode="Markdown")

# ------------------------------------------------------------------ #
#                            BOT COMMANDS                            #
# ------------------------------------------------------------------ #
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Hi! Upload a `.pdf`, `.docx`, or `.md` file, then ask questions "
        "about it — or type /help for commands."
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📘 *Help Menu*\n\n"
        "- Upload a PDF, DOCX, or MD file\n"
        "- Ask questions based on uploaded content\n"
        "- Use 'prepare me for exam' for notes/flashcards\n"
        "- Say 'more flashcards' for additional ones\n"
        "- Ask 'give me the answer in X points' for point summaries\n"
        "- /reset to clear uploaded data\n"
        "- /export to download your flashcards as PDF",
        parse_mode="Markdown",
    )

async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global passages, page_passages, index, all_flashcards
    passages, page_passages, all_flashcards = [], [], []
    index = faiss.IndexFlatL2(EMBED_DIM)
    await update.message.reply_text("🧹 Reset complete. Upload a document to start again.")

async def export_flashcards(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if all_flashcards:
        stream = generate_flashcards_pdf()
        await update.message.reply_document(InputFile(stream, filename="flashcards.pdf"))
    else:
        await update.message.reply_text("⚠️ No flashcards to export yet.")

# ------------------------------------------------------------------ #
#                       FILE & QUERY HANDLERS                        #
# ------------------------------------------------------------------ #
async def handle_file(update: Update, context: ContextTypes.DEFAULT_TYPE):
    document: Document = update.message.document
    path = os.path.join(DOC_DIR, document.file_name)

    tele_file = await context.bot.get_file(document.file_id)
    await tele_file.download_to_drive(path)

    text_pages = extract_text(path)
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
    await update.message.reply_text(
        f"✅ Document '{document.file_name}' indexed.\n\n📝 Summary Preview:"
    )
    await send_summary_chunks(preview, update)

# ---------------------   QUERY   ---------------------------------- #
async def handle_query(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.message.text.lower().strip()
    user_id = update.message.from_user.id

    # ---------- exam mode ----------
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

    # ---------- page‑specific ----------
    page_match = re.search(r"page\s+(\d+)", query)
    if page_match:
        num = int(page_match.group(1))
        if 1 <= num <= len(page_passages):
            await send_summary_chunks(
                summarize_passages(page_passages[num - 1], query=query), update
            )
        else:
            await update.message.reply_text("⚠️ Page number out of range.")
        return

    # ---------- “in N points” ----------
    points_match = re.search(r"in\s+(\d+)\s+points?", query)
    num_points = int(points_match.group(1)) if points_match else 30

    # ------------------------------------------------------------------ #
    # 1️⃣  NO DOCUMENTS YET  →  answer directly with Ollama LLM
    # ------------------------------------------------------------------ #
    if index.ntotal == 0:
        llm_answer = call_local_llm(query)
        await update.message.reply_text(llm_answer[:3900] if llm_answer else
                                        "⚠️ No documents uploaded and LLM failed.")
        return

    # ---------------- vector search ----------------
    query_emb = model.encode([query])
    D, I = index.search(query_emb, min(len(passages), TOP_K * 3))
    matched = [passages[i] for i in I[0] if 0 <= i < len(passages)]
    if not matched:
        matched = passages[:TOP_K * 3]

    summary = summarize_passages(matched, query=query, max_points=num_points)

    # ------------------------------------------------------------------ #
    # 2️⃣  LLM FALLBACK  when extractive summary is weak
    # ------------------------------------------------------------------ #
    if (
        not summary
        or summary == ["⚠️ Not enough content to summarize."]
        or len(" ".join(summary)) < 80
    ):
        context_snippet = "\n\n".join(matched[:3])       # ≤ ~750 words
        llm_answer = call_local_llm(
            prompt=f"{context_snippet}\n\nQuestion: {query}\n\nAnswer clearly:",
            system="You are an expert tutor."
        )
        if llm_answer:
            await update.message.reply_text(llm_answer[:3900])
            return
        # if Ollama fails, fall back to whatever summary we have

    # ---------- send extractive summary ----------
    await send_summary_chunks(summary, update)

# ------------------------------------------------------------------ #
#                               MAIN                                 #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    load_vector_store()
    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(CommandHandler("start",   start))
    app.add_handler(CommandHandler("help",    help_command))
    app.add_handler(CommandHandler("reset",   reset))
    app.add_handler(CommandHandler("export",  export_flashcards))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_file))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_query))

    print("🤖 Bot is running…")
    app.run_polling()
