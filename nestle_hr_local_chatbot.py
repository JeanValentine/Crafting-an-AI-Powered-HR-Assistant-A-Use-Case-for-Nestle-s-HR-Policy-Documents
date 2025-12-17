import argparse
from pathlib import Path
import os
import re
import json
import math
from typing import List, Tuple
from tqdm import tqdm

from pypdf import PdfReader

from sentence_transformers import SentenceTransformer
import numpy as np

import chromadb
from chromadb.config import Settings

import gradio as gr

CHROMA_DIR = "chroma_db_local"
COLLECTION_NAME = "nestle_hr_policy_local"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

def extract_text_from_pdf(pdf_path: str) -> List[Tuple[int, str]]:
    reader = PdfReader(pdf_path)
    pages = []
    for i, p in enumerate(reader.pages):
        try:
            text = p.extract_text() or ""
        except Exception:
            text = ""
        pages.append((i + 1, text))
    return pages

def chunk_texts(pages: List[Tuple[int, str]], chunk_size: int = 900, overlap: int = 150) -> List[dict]:
   
    chunks = []
    for page_num, text in pages:
        if not text:
            continue
    
        text = re.sub(r"\s+", " ", text).strip()
      
        sents = re.split(r'(?<=[.!?])\s+', text)
        cur = []
        cur_len = 0
        for s in sents:
            if cur_len + len(s) + 1 <= chunk_size or not cur:
                cur.append(s)
                cur_len += len(s) + 1
            else:
                chunk_text = " ".join(cur).strip()
                chunks.append({"page": page_num, "text": chunk_text})
               
                if overlap > 0:
                  
                    tail = chunk_text[-overlap:]
                    cur = [tail, s]
                    cur_len = len(tail) + len(s) + 1
                else:
                    cur = [s]
                    cur_len = len(s) + 1
        if cur:
            chunk_text = " ".join(cur).strip()
            chunks.append({"page": page_num, "text": chunk_text})
    return chunks

def _ensure_dir(d: str):
    Path(d).mkdir(parents=True, exist_ok=True)

def save_local_vectorstore(persist_dir: str, embeddings: np.ndarray, texts: List[str], metadatas: List[dict], ids: List[str]):
    _ensure_dir(persist_dir)
 
    np.savez_compressed(Path(persist_dir) / "embeddings.npz", embeddings=embeddings.astype(np.float32))
    with open(Path(persist_dir) / "texts.json", "w", encoding="utf-8") as f:
        json.dump(texts, f, ensure_ascii=False)
    with open(Path(persist_dir) / "metadatas.json", "w", encoding="utf-8") as f:
        json.dump(metadatas, f, ensure_ascii=False)
    with open(Path(persist_dir) / "ids.json", "w", encoding="utf-8") as f:
        json.dump(ids, f, ensure_ascii=False)
    print("Saved simple local vector store to", persist_dir)

def load_local_vectorstore(persist_dir: str):
    emb_path = Path(persist_dir) / "embeddings.npz"
    if not emb_path.exists():
        raise FileNotFoundError(f"No local vector store found at {persist_dir}")
    data = np.load(emb_path)
    embeddings = data["embeddings"]
    with open(Path(persist_dir) / "texts.json", "r", encoding="utf-8") as f:
        texts = json.load(f)
    with open(Path(persist_dir) / "metadatas.json", "r", encoding="utf-8") as f:
        metadatas = json.load(f)
    with open(Path(persist_dir) / "ids.json", "r", encoding="utf-8") as f:
        ids = json.load(f)
    return embeddings, texts, metadatas, ids

def query_local_store(embeddings: np.ndarray, texts: List[str], metadatas: List[dict], ids: List[str], query_emb: np.ndarray, k: int = 5):

    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)
    q = query_emb.astype(np.float32)

    emb_norms = np.linalg.norm(embeddings, axis=1)
    q_norm = np.linalg.norm(q)
    if q_norm == 0 or np.any(emb_norms == 0):
        sims = embeddings @ q
    else:
        sims = (embeddings @ q) / (emb_norms * q_norm)
   
    idx = np.argsort(-sims)[:k]
    results = []
    for i in idx:
        results.append({"id": ids[i], "text": texts[i], "meta": metadatas[i], "score": float(sims[i])})
    return results

def build_chroma_local(pdf_path: str, persist_dir: str = CHROMA_DIR, model_name: str = EMBED_MODEL_NAME):
    pdf_path = Path(pdf_path)
    assert pdf_path.exists(), f"PDF not found: {pdf_path}"
    print("Extracting text from PDF...")
    pages = extract_text_from_pdf(str(pdf_path))
    print(f"Extracted {len(pages)} pages.")
    print("Chunking text...")
    chunks = chunk_texts(pages, chunk_size=900, overlap=150)
    print(f"Created {len(chunks)} chunks.")

    print("Loading embedding model:", model_name)
    model = SentenceTransformer(model_name)

    texts = [c["text"] for c in chunks]
    metadatas = [{"page": c["page"]} for c in chunks]
    ids = [f"chunk_{i}" for i in range(len(texts))]

    print("Computing embeddings (this may take a minute)...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    embeddings = np.array(embeddings, dtype=np.float32)
   
    try:
        print("Attempting to create Chroma client and persist DB in", persist_dir)
        settings = Settings(chroma_db_impl="duckdb+parquet", persist_directory=persist_dir)
        client = chromadb.Client(settings=settings)

        try:
            client.get_collection(name=COLLECTION_NAME)
            client.delete_collection(name=COLLECTION_NAME)
        except Exception:
            pass

        collection = client.create_collection(name=COLLECTION_NAME)
        collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings.tolist()
        )
        client.persist()
        print("Chroma DB built and persisted.")
        return {"type": "chroma", "persist_dir": persist_dir}
    except Exception as e:
       
        print("Chroma client creation failed (will use simple local store instead).")
        print("Chroma error:", e)
      
        save_local_vectorstore(persist_dir, embeddings, texts, metadatas, ids)
        return {"type": "local", "persist_dir": persist_dir, "embeddings_shape": embeddings.shape}

def load_vectorstore(persist_dir: str = CHROMA_DIR):
 
    try:
        settings = Settings(chroma_db_impl="duckdb+parquet", persist_directory=persist_dir)
        client = chromadb.Client(settings=settings)
        collection = client.get_collection(name=COLLECTION_NAME)
        return {"type": "chroma", "client": client, "collection": collection}
    except Exception as e:
       
        try:
            embeddings, texts, metadatas, ids = load_local_vectorstore(persist_dir)
            return {"type": "local", "embeddings": embeddings, "texts": texts, "metadatas": metadatas, "ids": ids}
        except Exception as e2:
      
            raise RuntimeError(f"No usable vector store found at {persist_dir}. Chroma err: {e}; Local err: {e2}")

def query_collection(store, model, query: str, k: int = 5):
    q_emb = model.encode([query], convert_to_numpy=True)[0]
    if store["type"] == "chroma":
        collection = store["collection"]
        q_emb_list = q_emb.tolist()
        resp = collection.query(query_embeddings=[q_emb_list], n_results=k, include=["documents","metadatas","distances","ids"])
        results = []
        docs = resp.get("documents", [[]])[0]
        metas = resp.get("metadatas", [[]])[0]
        dists = resp.get("distances", [[]])[0]
        ids = resp.get("ids", [[]])[0]
        for doc, meta, dist, id_ in zip(docs, metas, dists, ids):
            results.append({"id": id_, "text": doc, "meta": meta, "distance": dist})
        return results
    else:
 
        embeddings = store["embeddings"]
        texts = store["texts"]
        metadatas = store["metadatas"]
        ids = store["ids"]
        return query_local_store(embeddings, texts, metadatas, ids, q_emb, k=k)

def serve_gradio_local(persist_dir: str = CHROMA_DIR, model_name: str = EMBED_MODEL_NAME, host="127.0.0.1", port=7860):

    print("Loading embedding model:", model_name)
    model = SentenceTransformer(model_name)
    print("Loading vector store from", persist_dir)
    store = load_vectorstore(persist_dir)

    def answer_fn(user_query, history):
        if not user_query:
            return "", history
        try:
            hits = query_collection(store, model, user_query, k=5)
        except Exception as e:
            return f"Error querying vector store: {e}", history

        if not hits:
            reply = "No relevant information found in the document."
        else:
       
            pieces = []
            for h in hits:
                page = h.get("meta", {}).get("page", "n/a")
                text = re.sub(r"\s+", " ", h.get("text", "")).strip()
         
                score = h.get("score") or h.get("distance")
                if score is not None:
                    pieces.append(f"[page:{page}] {text[:800]}  (score: {round(float(score), 4)})")
                else:
                    pieces.append(f"[page:{page}] {text[:800]}")
            reply = "\n\n---\n\n".join(pieces)
        history = history + [(user_query, reply)]
        return "", history

    with gr.Blocks() as demo:
        gr.Markdown("# Nestlé HR Policy — Local Retriever Chat")
        gr.Markdown("This local-only app retrieves the most relevant policy snippets (no LLM).")
        chatbot = gr.Chatbot()
        txt = gr.Textbox(placeholder="Ask about the HR policy (e.g., 'maternity leave')", show_label=False)
        state = gr.State([])
        txt.submit(answer_fn, [txt, state], [txt, chatbot])
        clear = gr.Button("Clear")
        clear.click(lambda: (None, []), None, [txt, state], queue=False)

    demo.launch(server_name=host, server_port=port, share=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=str, default="the_nestle_hr_policy_pdf_2012.pdf")
    parser.add_argument("--build_db", action="store_true")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--persist_dir", type=str, default=CHROMA_DIR)
    args = parser.parse_args()

    if args.build_db:
        build_chroma_local(args.pdf, persist_dir=args.persist_dir)
    if args.serve:
   
        if not Path(args.persist_dir).exists():
            raise SystemExit("Persist directory not found. Run with --build_db first.")
        serve_gradio_local(persist_dir=args.persist_dir)

if __name__ == "__main__":
    main()