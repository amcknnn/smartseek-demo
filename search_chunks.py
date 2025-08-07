import openai
import os
import pandas as pd
import numpy as np

# 🔐 Load your API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# 📄 Load the data
df = pd.read_csv("data/video_chunks.csv")

# 🧠 Convert string embeddings back to numpy arrays
df["embedding"] = df["embedding"].apply(lambda x: np.array(eval(x)))

# 🔍 Function to get embedding for a question
def get_embedding(text, model="text-embedding-3-small"):
    response = openai.embeddings.create(
        input=text,
        model=model
    )
    return np.array(response.data[0].embedding)

# 🎯 Function to search
def search_videos(query, top_k=5):
    query_embedding = get_embedding(query)
    df["similarity"] = df["embedding"].apply(lambda x: np.dot(x, query_embedding))
    top_results = df.sort_values("similarity", ascending=False).head(top_k)
    return top_results[["source_file", "text", "start_time", "video_link", "similarity"]]

# 💬 Main
if __name__ == "__main__":
    while True:
        query = input("\n🔎 Ask your question (or type 'exit'): ")
        if query.lower() == "exit":
            break

        results = search_videos(query)

        print("\n📽️ Top matches:\n")
        for i, row in results.iterrows():
            print(f"🎬 File: {row['source_file']}")
            print(f"📝 Snippet: {row['text'][:200]}...")
            print(f"⏱️ Timestamp: {row['start_time']}s")
            print(f"🔗 Link: {row['video_link']}")
            print(f"⭐ Similarity: {round(row['similarity'], 3)}")
            print("-" * 60)
