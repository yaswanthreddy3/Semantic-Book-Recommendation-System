import os
import math
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, url_for

# LangChain loaders / splitters
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

# Use langchain-chroma (future-proof)
from langchain_chroma import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

app = Flask(__name__)

# ---------- CONFIG ----------
BOOKS_CSV = "books_with_emotion_scores.csv"   # place your CSV here
TAGGED_DESC = "tagged_description.txt"        # place your tagged descriptions here
PERSIST_DIR = "book_db"                       # local chroma DB folder
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # embedding model
HOST = "0.0.0.0"
PORT = 7860
DEBUG = True

# ---------- LOAD BOOKS ----------
if not os.path.exists(BOOKS_CSV):
    raise FileNotFoundError(f"{BOOKS_CSV} not found. Add your CSV file to the project root.")

books = pd.read_csv(BOOKS_CSV)
if 'isbn13' not in books.columns:
    raise ValueError("books CSV must have an 'isbn13' column (numeric or string).")

books['thumbnail'] = books.get('thumbnail', pd.Series()).fillna('image.png')
books['large_thumbnail'] = np.where(
    books['thumbnail'] != 'image.png',
    books['thumbnail'].astype(str) + '&fife=w800',
    'image.png'
)

CATEGORIES = ['All'] + sorted(map(str, books['simple_categories'].dropna().unique()))
TONES = ['All', 'Happy', 'Surprising', 'Angry', 'Suspenseful', 'Sad']

# ---------- EMBEDDINGS & CHROMA ----------
os.makedirs(PERSIST_DIR, exist_ok=True)
embedding = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

def _exists_chroma_dir(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    names = set(os.listdir(path))
    markers = {
        "chroma.sqlite3",
        "index",
        "chroma-collections.parquet",
        "chroma-embeddings.parquet",
    }
    return len(names.intersection(markers)) > 0

def build_chroma_from_tagged_text():
    if not os.path.exists(TAGGED_DESC):
        raise FileNotFoundError(f"{TAGGED_DESC} not found. Provide tagged_description.txt to build DB.")
    raw_documents = TextLoader(TAGGED_DESC).load()
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1, chunk_overlap=0)
    documents = text_splitter.split_documents(raw_documents)
    db = Chroma.from_documents(documents=documents, embedding=embedding, persist_directory=PERSIST_DIR)
    try:
        db.persist()
    except Exception:
        pass
    return db

def load_or_create_db():
    if not _exists_chroma_dir(PERSIST_DIR):
        print("📌 Creating new Chroma database from tagged_description.txt ...")
        return build_chroma_from_tagged_text()
    else:
        print("📌 Loading existing Chroma database...")
        return Chroma(persist_directory=PERSIST_DIR, embedding_function=embedding)

db_books = load_or_create_db()

# ---------- Recommendation core ----------
TONE_FIELD = {
    "Happy": "joy",
    "Surprising": "surprise",
    "Angry": "anger",
    "Suspenseful": "fear",
    "Sad": "sadness"
}

def retrieve_semantic_recommendation(
    query: str,
    category: str = 'All',
    tone: str = 'All',
    initial_top_k: int = 50,
    sort_top_k: int = 32
) -> pd.DataFrame:
    if not query or not query.strip():
        return pd.DataFrame(columns=books.columns)

    recs = db_books.similarity_search_with_score(query.strip(), k=initial_top_k)
    idxs = []
    for doc, _score in recs:
        text = doc.page_content.strip()
        parts = text.split()
        head = parts[0].strip().strip('"').strip("'") if parts else ""
        try:
            idxs.append(int(head))
        except Exception:
            continue

    rec_df = books[books['isbn13'].isin(idxs)].copy()
    if rec_df.empty and len(recs) > 0:
        candidates = []
        for doc, _score in recs[:initial_top_k]:
            txt = doc.page_content.strip()
            parts = txt.split()
            if parts and parts[0].isdigit():
                snippet = " ".join(parts[1:6])
            else:
                snippet = " ".join(parts[:6])
            candidates.append(snippet.lower())

        matched_isbns = set()
        for snip in candidates:
            words = [w for w in snip.split() if len(w) > 2]
            if not words:
                continue
            for i in range(max(0, len(words)-1)):
                seq = f"{words[i]} {words[i+1]}"
                mask_title = books['title'].str.lower().str.contains(seq, na=False)
                mask_auth = books['authors'].str.lower().str.contains(seq, na=False)
                found = books.loc[mask_title | mask_auth, 'isbn13'].tolist()
                for f in found:
                    matched_isbns.add(f)
            if len(matched_isbns) >= sort_top_k:
                break
        if matched_isbns:
            rec_df = books[books['isbn13'].isin(list(matched_isbns))].copy()

    if not rec_df.empty and category != 'All':
        rec_df = rec_df[rec_df['simple_categories'] == category]

    if not rec_df.empty and tone in TONE_FIELD:
        field = TONE_FIELD[tone]
        if field in rec_df.columns:
            rec_df[field] = pd.to_numeric(rec_df[field], errors='coerce').fillna(0.0)
            rec_df = rec_df.sort_values(by=field, ascending=False)

    if rec_df.empty:
        if 'average_rating' in books.columns:
            fallback = books.copy()
            fallback['average_rating'] = pd.to_numeric(fallback['average_rating'], errors='coerce').fillna(0.0)
            rec_df = fallback.sort_values(by='average_rating', ascending=False).head(sort_top_k)
        else:
            rec_df = books.head(sort_top_k).copy()

    return rec_df.head(sort_top_k).reset_index(drop=True)

def format_author_list(authors: str) -> str:
    parts = [p.strip() for p in str(authors).split(',') if p.strip()]
    if len(parts) == 0:
        return "Unknown"
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return f"{', '.join(parts[:-1])} and {parts[-1]}"

def truncate_text(text: str, words: int = 30) -> str:
    s = str(text) if pd.notnull(text) else ""
    toks = s.split()
    if len(toks) <= words:
        return s
    return " ".join(toks[:words]) + "..."

# ---------- Routes ----------
@app.route("/", methods=["GET"])
def index():
    query = request.args.get("query", "", type=str)
    category = request.args.get("category", "All", type=str)
    tone = request.args.get("tone", "All", type=str)
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 16, type=int)

    # Show top 50 books when no query
    if not query.strip():
        results_df = books.copy()
        if 'average_rating' in books.columns:
            results_df['average_rating'] = pd.to_numeric(results_df['average_rating'], errors='coerce').fillna(0.0)
            results_df = results_df.sort_values(by='average_rating', ascending=False)
        results_df = results_df.head(50)
    else:
        results_df = retrieve_semantic_recommendation(query, category, tone)

        # ---------------- Title-priority ----------------
        title_mask = books['title'].str.lower().str.contains(query.strip().lower(), na=False)
        if title_mask.any():
            title_df = books[title_mask].copy()
            results_df = pd.concat([title_df, results_df]).drop_duplicates(subset='isbn13').reset_index(drop=True)

    total = len(results_df)
    pages = max(1, math.ceil(total / per_page))
    page = max(1, min(page, pages))

    start = (page - 1) * per_page
    end = start + per_page
    page_df = results_df.iloc[start:end] if total > 0 else results_df

    cards = []
    for _, row in page_df.iterrows():
        caption = f"{row['title']} by {format_author_list(row['authors'])}"
        desc = truncate_text(row.get('description', ''), 36)
        rating = row.get('average_rating', None)
        pub_year = row.get('published_year', None)
        img = row.get('large_thumbnail') or 'image.png'
        if img == 'image.png':
            img = url_for('static', filename='image.png')

        cards.append({
            "image": img,
            "title": row['title'],
            "caption": caption,
            "desc": desc if desc else "No description available.",
            "rating": float(rating) if pd.notnull(rating) else None,
            "year": int(pub_year) if pd.notnull(pub_year) else None
        })

    sample_suggestions = list(books['title'].dropna().head(30).values)

    return render_template(
        "index.html",
        categories=CATEGORIES,
        tones=TONES,
        query=query,
        category=category,
        tone=tone,
        cards=cards,
        total=total,
        page=page,
        pages=pages,
        per_page=per_page,
        suggestions=sample_suggestions
    )

if __name__ == "__main__":
    app.run(host=HOST, port=PORT, debug=DEBUG)
