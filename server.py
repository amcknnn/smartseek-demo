from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import numpy as np
import os, ast
from openai import OpenAI

# ---- CONFIG ----
CSV_PATH   = "data/video_chunks.csv"
VIDEO_DIR  = "mp4"
TOP_K      = 5
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
df["embedding"] = df["embedding"].apply(ast.literal_eval)

def cosine_sim(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

@app.route("/search")
def search():
    query = request.args.get("query", "")
    if not query:
        return jsonify({"results": []})

    # Embed the query
    q_emb = client.embeddings.create(model=EMBED_MODEL, input=query).data[0].embedding

    # Similarity
    df["similarity"] = df["embedding"].apply(lambda e: cosine_sim(e, q_emb))
    top = df.sort_values("similarity", ascending=False).head(TOP_K)

    results = []
    for _, row in top.iterrows():
        results.append({
            "text": row["text"],
            "start_time": int(row["start_time"]),
            "end_time": int(row["end_time"]),
            "source_file": row["source_file"],
            # <- IMPORTANT: point straight to /mp4/<filename>
            "video_link": f"/mp4/{row['source_file']}"
        })
    return jsonify({"results": results})

@app.route("/mp4/<path:filename>")
def serve_video(filename):
    # Explicitly serve from mp4/ and with correct mimetype
    return send_from_directory(VIDEO_DIR, filename, mimetype="video/mp4")

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

if __name__ == "__main__":
    app.run(debug=True)
