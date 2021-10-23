#!/usr/bin/env python3
"""
Utility script to download all required NLTK resources for DeepFinNER.
Run once before first use (or include in deployment pipeline).
"""
import nltk

RESOURCES = [
    "punkt",
    "punkt_tab",  # fallback tokenizer tables
    "stopwords",
    "wordnet",
    "averaged_perceptron_tagger",
]

if __name__ == "__main__":
    for res in RESOURCES:
        print(f"Downloading {res} ...")
        nltk.download(res)
    print("✅ NLTK setup complete.") 