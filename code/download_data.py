# -*- coding: utf-8 -*-
"""Download alpaca-gpt4-data-zh dataset."""
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
from huggingface_hub import hf_hub_download

REPO = "llm-wizard/alpaca-gpt4-data-zh"
FILENAME = "alpaca_gpt4_data_zh.json"

def main():
    path = hf_hub_download(repo_id=REPO, filename=FILENAME, repo_type="dataset")
    print("downloaded to:", path)

if __name__ == "__main__":
    main()
