# ! pip install transformers
# ! pip install pytorch
# https://github.com/BM-K/KoSentenceBERT-ETRI

import numpy as np
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import argparse
import codecs
import os
import torch

# Set up argument parser
parser = argparse.ArgumentParser()
parser.add_argument("comments_file", help="Path to the YouTube comments txt file")
parser.add_argument("seed_file", help="Path to seed words txt file")
parser.add_argument("topn", type=int, default=5, help="Number of most similar words to include in the expanded dictionary")
parser.add_argument("--new_seed_file", default=None, help="Path to the new seed_file")
args = parser.parse_args()

# Load KoSentenceBERT-ETRI model and tokenizer
model_path = "KoSentenceBERT-ETRI"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertModel.from_pretrained(model_path)

# Function to get BERT embeddings
def get_bert_embedding(word):
    tokens = tokenizer(word, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**tokens)
    embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    return embedding

# Load YouTube comments and create a list of unique words
unique_words = set()
with codecs.open(args.comments_file, encoding="euc-kr") as f:
    for line in f:
        words = line.strip().split()
        for word in words:
            unique_words.add(word)

# Calculate BERT embeddings for all unique words
word_embeddings = {}
for word in unique_words:
    word_embeddings[word] = get_bert_embedding(word)

# Load seed words
with open(args.seed_file) as f:
    og_seeds = [line.strip() for line in f]

# Create expanded dictionary with the N most similar words for each seed word
expanded_dict = []

if args.new_seed_file:
    with open(args.new_seed_file) as f:
        seeds = [line.strip() for line in f]
else:
    with open(args.seed_file) as f:
        seeds = [line.strip() for line in f]

for seed in seeds:
    seed_embedding = get_bert_embedding(seed)
    similarities = {}
    for word in word_embeddings:
        similarity = cosine_similarity([seed_embedding], [word_embeddings[word]])
        similarities[word] = similarity[0][0]
    sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    expanded_dict.append(seed)
    for i in range(min(args.topn, len(sorted_similarities))):
        word, similarity = sorted_similarities[i]
        expanded_dict.append(word)

# Write expanded dictionary to output file
expanded_dict_filename = f'expanded_dict_{model_path}.txt'
with open(expanded_dict_filename, 'w') as outfile:
    outfile.write('\n'.join(expanded_dict))

# Get top N similar words for each seed word and write to output file
output_filename = f'output_{model_path}.txt'
with open(output_filename, 'w') as outfile:
    for seed in og_seeds:
        outfile.write(f"Top 5 similar words for '{seed}':\n")
        if seed not in expanded_dict:
            continue
        similarities = {}
        seed_embedding = get_bert_embedding(seed)
        for word in expanded_dict:
            if word == seed:
                continue
            word_embedding = get_bert_embedding(word)
            similarity = cosine_similarity([seed_embedding], [word_embedding])
            similarities[word] = similarity[0][0]
        sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        for i in range(min(5, len(sorted_similarities))):
            word, similarity = sorted_similarities[i]
            outfile.write(f"{word}: {similarity}\n")
        outfile.write("\n")
