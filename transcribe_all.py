import os
import subprocess

INPUT_FOLDER = "mp4"
OUTPUT_FOLDER = "transcripts"

# Skip files that are known to have no audio or shouldn't be processed
SKIP_FILES = [
    "SystemAttackTopPinsVol7.mp4"
]

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

for filename in os.listdir(INPUT_FOLDER):
    if not filename.endswith(".mp4"):
        continue

    if filename in SKIP_FILES:
        print(f"🚫 Skipping {filename} (manually excluded)")
        continue

    base_name = os.path.splitext(filename)[0]
    transcript_path = os.path.join(OUTPUT_FOLDER, base_name + ".txt")

    if os.path.exists(transcript_path):
        print(f"⏩ Skipping {filename} (already transcribed)")
        continue

    input_path = os.path.join(INPUT_FOLDER, filename)
    print(f"🎙️ Transcribing {filename}...")

    subprocess.run([
        "whisper", input_path,
        "--model", "base",
        "--language", "en",
        "--output_format", "txt",
        "--output_dir", OUTPUT_FOLDER
    ])

print("✅ Transcription complete.")
