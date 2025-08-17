from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import numpy as np
import os, ast
from openai import OpenAI

# ---------- CONFIG ----------
CSV_PATH     = "data/video_chunks.csv"
VIDEO_DIR    = "mp4"                      # kept for optional local serving
TOP_K        = 5
EMBED_MODEL  = "text-embedding-3-small"

# ---------- OPENAI ----------
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("Set OPENAI_API_KEY and restart your terminal.")
client = OpenAI(api_key=api_key)

# ---------- FLASK ----------
app = Flask(__name__)

# ---------- LOAD DATA ----------
df = pd.read_csv(CSV_PATH)

# Parse embeddings: stored as JSON strings; convert to list[float]
def _to_vec(x):
    if isinstance(x, (list, tuple)):
        return list(map(float, x))
    return list(map(float, ast.literal_eval(x)))

df["embedding"] = df["embedding"].apply(_to_vec)

# Decide which column holds the video URL (S3) vs. local path
LINK_COL = "video" if "video" in df.columns else ("video_link" if "video_link" in df.columns else None)
if LINK_COL is None:
    raise RuntimeError("CSV must contain 'video' or 'video_link' with full video URLs.")

def cosine_sim(a, b):
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0

# Serve files inside the data/ folder (e.g., /data/video_chunks.csv)
@app.route("/data/<path:filename>")
def serve_data(filename):
    return send_from_directory("data", filename)

@app.route("/search")
def search():
    query = request.args.get("query", "").strip()
    if not query:
        return jsonify({"results": []})

    # 1) Embed the query
    q_emb = client.embeddings.create(model=EMBED_MODEL, input=query).data[0].embedding

    # 2) Similarity (avoid mutating global df)
    sims = df["embedding"].apply(lambda e: cosine_sim(e, q_emb))
    scored = df.assign(similarity=sims)

    # 3) Top K
    top = scored.sort_values("similarity", ascending=False).head(TOP_K).copy()

    # 4) Clean rows to ensure JSON-safe output
    def good_link(val):
        return isinstance(val, str) and val.startswith(("http://", "https://"))

    top[LINK_COL] = top[LINK_COL].fillna("")
    top["source_file"] = top.get("source_file", "").fillna("")
    top = top[top[LINK_COL].apply(good_link)].copy()

    results = []
    for _, row in top.iterrows():
        try:
            results.append({
                "text": (row.get("text") or ""),
                "start_time": int(float(row.get("start_time", 0) or 0)),
                "end_time": int(float(row.get("end_time", 0) or 0)),
                "source_file": (row.get("source_file") or ""),
                "video_link": row[LINK_COL],                       # S3 (or full) URL from CSV
                "similarity": float(row.get("similarity", 0.0) or 0.0),
            })
        except Exception:
            # Skip any row that can't be coerced cleanly
            continue

    return jsonify({"results": results})

# Optional local serving (not used when you have S3 URLs)
@app.route("/mp4/<path:filename>")
def serve_video(filename):
    return send_from_directory(VIDEO_DIR, filename, mimetype="video/mp4")

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

if __name__ == "__main__":
    app.run(debug=True)
