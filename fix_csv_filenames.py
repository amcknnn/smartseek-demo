# fix_csv_filenames.py
import pandas as pd

# Load the CSV
df = pd.read_csv("data/video_chunks.csv")

# Strip out anything after ".mp4" (e.g., " (1080p_aac)")
df["source_file"] = df["source_file"].str.extract(r"(.*?\.mp4)")

# Save it back to the same file
df.to_csv("data/video_chunks.csv", index=False)

print("✅ Cleaned source_file column and saved.")
