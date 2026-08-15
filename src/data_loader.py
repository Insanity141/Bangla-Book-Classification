import os, shutil
from huggingface_hub import hf_hub_download

REPO_ID = "DevnilMaster1/Bangla-Book-Recommendation-Dataset"
DATA_FOLDER = "data/RokomariBG_Dataset"
os.makedirs(DATA_FOLDER, exist_ok=True)

FILES_NEEDED = [
    'book.json',
    'author.json',
    'category.json',
    'publisher.json',
    'review.json',
    'user_to_review.json',
    'book_to_review.json',
    'book_to_author.json',
    'book_to_category.json',
    'book_to_publisher.json'
    ]

for filename in FILES_NEEDED:
    dest = os.path.join(DATA_FOLDER, filename)
    if not os.path.exists(dest):
        try:
            downloaded_path = hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                repo_type="dataset"
            )
            shutil.copy(downloaded_path, dest)
            print(f"✅ Saved: {filename}")
        except Exception as e:
            print(f"⚠️ Error downloading {filename}: {e}")
    else:
        print(f"✅ Already present: {filename}")


import json

def load_json_file(filename):

    path = os.path.join(DATA_FOLDER, filename)

    if not os.path.exists(path):
        print(f"File not found: {filename}")
        return None

    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)

    except json.JSONDecodeError:

        with open(path, "r", encoding="utf-8") as file:
            return [
                json.loads(line)
                for line in file
                if line.strip()
            ]

    loaded_data = {}

    for filename in FILES_NEEDED:

        data = load_json_file(filename)

        if data is not None:
            loaded_data[filename] = data

            print(f"{filename}: {len(data)} records")
