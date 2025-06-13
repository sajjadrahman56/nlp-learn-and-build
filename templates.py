import os

# Use current directory as root
ROOT = "."

structure = {
    "0-roadmap": ["0-nlp-roadmap.md"],
    "1-data-formats": ["read_csv_parquet.ipynb", "spark_ingestion.py"],
    "2-text-cleaning": ["text_cleaning_utils.py", "cleaning_pipeline.ipynb"],
    "3-feature-engineering": ["tfidf_word2vec.ipynb"],
    "4-classical-ml": ["product_review_classifier.ipynb"],
    "5-transformers-llm": ["bengali_bert_sentiment.ipynb", "llm_finetune_lora.ipynb"],
    "6-deployment": ["app.py", "dockerfile"],
    "7-capstone-projects": ["resume_screener", "bengali_chatbot"],
    "assets": ["data_formats_diagram.png"],
}

root_files = ["requirements.txt", ".gitignore", "README.md"]

def create_structure(base_path=ROOT):
    for folder, contents in structure.items():
        folder_path = os.path.join(base_path, folder)
        os.makedirs(folder_path, exist_ok=True)
        print(f"Created folder: {folder_path}")

        for item in contents:
            item_path = os.path.join(folder_path, item)
            if "." in item:
                if not os.path.exists(item_path):
                    with open(item_path, "w") as f:
                        if item.endswith(".md"):
                            f.write(f"# {item.replace('_', ' ').replace('.md', '').title()}\n")
                        elif item.endswith(".py"):
                            f.write(f"# {item.replace('_', ' ').replace('.py', '').title()}\n")
                        elif item.endswith(".ipynb"):
                            f.write('{\n "cells": [],\n "metadata": {},\n "nbformat": 4,\n "nbformat_minor": 4\n}')
                    print(f"Created file: {item_path}")
            else:
                os.makedirs(item_path, exist_ok=True)
                print(f"Created subfolder: {item_path}")

    for file in root_files:
        file_path = os.path.join(base_path, file)
        if not os.path.exists(file_path):
            with open(file_path, "w") as f:
                if file == "README.md":
                    f.write("# NLP Engineering Lab\n\nProject repository for my NLP roadmap.\n")
                else:
                    f.write("")
            print(f"Created root file: {file_path}")

if __name__ == "__main__":
    create_structure()
