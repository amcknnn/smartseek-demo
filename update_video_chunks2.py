import pandas as pd

df = pd.read_csv("data/video_chunks.csv")
df.rename(columns={"video_link": "video"}, inplace=True)
df.to_csv("data/video_chunks.csv", index=False)
