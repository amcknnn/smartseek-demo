import os
import urllib.parse
import pandas as pd

CSV = "data/video_chunks.csv"
BUCKET = "smartseek-demo-videos-amcknnn"   # <-- your bucket name
BASE = f"https://{BUCKET}.s3.amazonaws.com/videos/"

df = pd.read_csv(CSV)

def to_s3(url_or_path: str) -> str:
    """
    Accepts something like:
      'static/videos/HalfGuard (1080p).mp4#t=120'
    and returns:
      'https://<bucket>.s3.amazonaws.com/videos/HalfGuard%20(1080p).mp4#t=120'
    """
    if not isinstance(url_or_path, str):
        return url_or_path

    # Split off timestamp fragment if present
    if "#t=" in url_or_path:
        base, frag = url_or_path.split("#", 1)
        frag = "#" + frag
    else:
        base, frag = url_or_path, ""

    # Only keep the filename
    fname = os.path.basename(base)

    # URL-encode filename (leave . _ - ( ) safe, encode spaces)
    fname_enc = urllib.parse.quote(fname, safe="._-() ").replace(" ", "%20")

    return f"{BASE}{fname_enc}{frag}"

# Use the correct column name
if "video_link" not in df.columns:
    raise RuntimeError(f"'video_link' column not found in {CSV}. Columns: {list(df.columns)}")

df["video_link"] = df["video_link"].apply(to_s3)
df.to_csv(CSV, index=False)

print(f"✅ Updated {CSV} video_link → S3 URLs")
