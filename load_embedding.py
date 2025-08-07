import csv
import ast  # To safely convert stringified lists to real lists

# Path to your saved CSV
csv_path = "HalfGuard_embeddings.csv"

# Store the data
chunks = []
embeddings = []

with open(csv_path, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        chunks.append(row["text"])
        # Convert stringified list to real list of floats
        embedding_vector = ast.literal_eval(row["embedding"])
        embeddings.append(embedding_vector)

print(f"✅ Loaded {len(chunks)} chunks and their embeddings.")