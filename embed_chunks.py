import os
import csv
import json
from uuid import uuid4
from pathlib import Path

from openai import OpenAI
import tiktoken

# =============== CONFIG ===============
TRANSCRIPTS_DIR = Path("transcripts")
DATA_DIR = Path("data")
OUTPUT_CSV = DATA_DIR / "video_chunks.csv"

# Embedding model (newer, cheaper than ada-002)
EMBED_MODEL = "text-embedding-3-small"

# Chunking settings (word-based for simplicity)
CHUNK_WORDS = 400          # ~ how many words per chunk
OVERLAP_WORDS = 50         # overlap to keep context
WORDS_PER_SECOND = 2.5     # rough estimate to compute timestamps

# =====================================

def require_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. In a new terminal, run:\n\n"
            '  setx OPENAI_API_KEY "YOUR_API_KEY_HERE"\n\n'
            "Then close & reopen the terminal and try again."
        )
    return api_key

def load_already_done(csv_path: Path):
    """Return a set of source_file names already in the CSV so we can skip them."""
    done = set()
    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                done.add(row["source_file"])
    return done

def chunk_with_timestamps(text: str,
                          chunk_words: int = CHUNK_WORDS,
                          overlap_words: int = OVERLAP_WORDS,
                          wps: float = WORDS_PER_SECOND):
    """
    Split transcript text into overlapping word chunks and estimate start/end seconds.
    Returns a list of dicts: {chunk_index, text, start_time, end_time}
    """
    words = text.split()
    chunks = []
    i = 0
    idx = 0
    while i < len(words):
        chunk_w = words[i:i + chunk_words]
        if not chunk_w:
            break

        start_word = i
        end_word = i + len(chunk_w)

        start_time = int(start_word / wps)
        end_time = int(end_word / wps)

        chunks.append({
            "chunk_index": idx,
            "text": " ".join(chunk_w),
            "start_time": start_time,
            "end_time": end_time
        })

        idx += 1
        i += chunk_words - overlap_words  # slide forward with overlap

    return chunks

def main():
    api_key = require_api_key()
    client = OpenAI(api_key=api_key)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # If the CSV doesn't exist yet, we'll write the header.
    write_header = not OUTPUT_CSV.exists()
    processed_files = load_already_done(OUTPUT_CSV)

    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as out_f:
        fieldnames = [
            "id",
            "source_file",
            "chunk_index",
            "text",
            "embedding",
            "start_time",
            "end_time",
            "video_link"
        ]
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for txt_path in TRANSCRIPTS_DIR.glob("*.txt"):
            # Skip any non-whisper or junk files if present
            if "embedding" in txt_path.name.lower():
                continue

            if txt_path.name in processed_files:
                print(f"⏩ Skipping {txt_path.name} (already embedded)")
                continue

            print(f"🧠 Processing {txt_path.name}...")
            try:
                text = txt_path.read_text(encoding="utf-8")
            except Exception as e:
                print(f"❌ Could not read {txt_path.name}: {e}")
                continue

            # Build video link base name (strip .txt, add .mp4)
            base_name = txt_path.stem  # filename without extension
            video_filename = f"{base_name}.mp4"

            chunks = chunk_with_timestamps(text)

            for ch in chunks:
                try:
                    resp = client.embeddings.create(
                        model=EMBED_MODEL,
                        input=ch["text"]
                    )
                    embedding = resp.data[0].embedding

                    writer.writerow({
                        "id": str(uuid4()),
                        "source_file": txt_path.name,
                        "chunk_index": ch["chunk_index"],
                        "text": ch["text"],
                        "embedding": json.dumps(embedding),
                        "start_time": ch["start_time"],
                        "end_time": ch["end_time"],
                        "video_link": f"{video_filename}#t={ch['start_time']}"
                    })
                except Exception as e:
                    print(f"❌ Error embedding chunk {ch['chunk_index']} of {txt_path.name}: {e}")

    print(f"✅ Done. Saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
