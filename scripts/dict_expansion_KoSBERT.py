import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import argparse
import codecs
import os
import torch

# Set up argument parser
parser = argparse.ArgumentParser()
parser.add_argument("comments_file", help="Path to the YouTube comments txt file")
parser.add_argument("seed_file", help="Path to seed words txt file")
parser.add_argument("morpheme_file", help="Path to the morpheme txt file")
parser.add_argument("topn", type=int, default=5, help="Number of most similar words to include in the expanded dictionary")
parser.add_argument("--new_seed_file", default=None, help="Path to the new seed_file")
parser.add_argument("model_path", help= "path to model files")
args = parser.parse_args()

# Load KoSimCSE-bert model and tokenizer
model_path = args.model_path
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)

# Function to get BERT embeddings
def get_bert_embedding(word):
    tokens = tokenizer(word, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**tokens)
    embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    return embedding

# Calculate BERT embeddings for morphemes from the prepared .txt file
morpheme_dict = {}
morpheme_embeddings = {}
with codecs.open(args.morpheme_file, encoding="euc-kr") as f:
    for line in f:
        line = line.strip().split()
        morpheme = line[0]
        embeddings = []
        for word in morpheme.split("+"):
            embeddings.append(get_bert_embedding(word))
        if len(embeddings) > 0:
            morpheme_embeddings[morpheme] = np.mean(embeddings, axis=0)

# Load seed words
with open(args.seed_file) as f:
    seeds = [line.strip() for line in f]

# Find the most similar words to each seed word using morpheme embeddings
output_filename = f'output_{model_path}.txt'
with open(output_filename, 'w') as outfile:
    for seed in seeds:
        outfile.write(f"Top {args.topn} similar morphemes for '{seed}':\n")
        seed_embedding = get_bert_embedding(seed)
        similarities = {}
        for morpheme in morpheme_embeddings:
            similarity = cosine_similarity([seed_embedding], [morpheme_embeddings[morpheme]])
            similarities[morpheme] = similarity[0][0]
        sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        for i in range(min(args.topn, len(sorted_similarities))):
            morpheme, similarity = sorted_similarities[i]
            outfile.write(f"{morpheme}: {similarity}\n")
        outfile.write("\n")
