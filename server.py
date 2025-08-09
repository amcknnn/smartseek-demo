from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import numpy as np
import os, ast
from openai import OpenAI

# ---- CONFIG ----
CSV_PATH    = "data/video_chunks.csv"
VIDEO_DIR   = "mp4"                     # unused for S3 links but kept for local dev
TOP_K       = 5
EMBED_MODEL = "text-embedding-3-small"

# ---- OPENAI ----
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("Set OPENAI_API_KEY and restart your terminal.")
client = OpenAI(api_key=api_key)

# ---- FLASK ----
app = Flask(__name__)

# ---- LOAD DATA ----
df = pd.read_csv(CSV_PATH)

# Parse embeddings column from JSON string -> list[float]
df["embedding"] = df["embedding"].apply(lambda x: x if isinstance(x, (list, tuple)) else ast.literal_eval(x))

# Decide which column holds the video URL (S3) vs. local path
LINK_COL = "video" if "video" in df.columns else ("video_link" if "video_link" in df.columns else None)
if LINK_COL is None:
    raise RuntimeError(
        "Expected a 'video' or 'video_link' column in data/video_chunks.csv with full S3 URLs."
    )

def cosine_sim(a, b):
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0

@app.route("/search")
def search():
    query = request.args.get("query", "").strip()
    if not query:
        return jsonify({"results": []})

    # 1) Embed the query
    q_emb = client.embeddings.create(model=EMBED_MODEL, input=query).data[0].embedding

    # 2) Similarity per row
    sims = df["embedding"].apply(lambda e: cosine_sim(e, q_emb))
    top = df.assign(similarity=sims).sort_values("similarity", ascending=False).head(TOP_K)

    # 3) Build results, using the S3 URL from CSV
    results = []
    for _, row in top.iterrows():
        link = row[LINK_COL]
        results.append({
            "text": row["text"],
            "start_time": int(row["start_time"]),
            "end_time": int(row["end_time"]),
            "source_file": row["source_file"],
            "video_link": link,                # <- use S3 URL directly
            "similarity": float(row["similarity"]),
        })
    return jsonify({"results": results})

# Optional: local static serving if you ever use /mp4/<file> again
@app.route("/mp4/<path:filename>")
def serve_video(filename):
    return send_from_directory(VIDEO_DIR, filename, mimetype="video/mp4")

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

if __name__ == "__main__":
    app.run(debug=True)
