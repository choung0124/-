import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sklearn.metrics.pairwise import cosine_similarity
import argparse
import codecs
import os
from tqdm import tqdm
import torch

# Set up argument parser
parser = argparse.ArgumentParser()
parser.add_argument("morpheme_file", help="Path to morpheme txt file")
parser.add_argument("seed_file", help="Path to seed words txt file")
parser.add_argument("topn", type=int, default=5, help="Number of most similar words to include in the expanded dictionary")
parser.add_argument("--new_seed_file", default=None, help="Path to the new seed_file")
args = parser.parse_args()

# Load KoGPT2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("taeminlee/kogpt2")
model = GPT2LMHeadModel.from_pretrained("taeminlee/kogpt2", output_hidden_states=True)

# Function to get GPT2 embeddings
def get_gpt2_embeddings(words):
    tokens = tokenizer(words, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**tokens)
    embeddings = outputs.hidden_states[-1][:, 0, :].numpy()
    return embeddings

# Load morpheme dictionary and calculate embeddings for all morphemes
morpheme_dict = {}
morpheme_embeddings = {}

with codecs.open(args.morpheme_file, encoding="euc-kr") as f:
    lines = f.readlines()
    batch_size = 1000
    for i in tqdm(range(0, len(lines), batch_size), desc='Calculating embeddings'):
        batch_lines = lines[i:i+batch_size]
        morphemes = [line.strip().split()[0] for line in batch_lines]
        embeddings = get_gpt2_embeddings(morphemes)
        for morpheme, embedding in zip(morphemes, embeddings):
            morpheme_embeddings[morpheme] = embedding

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

for seed in tqdm(seeds, desc='Expanding dictionary'):
    seed_embedding = get_gpt2_embedding(seed)
    distances = {}
    for morpheme in morpheme_embeddings:
        distance = np.linalg.norm(seed_embedding - morpheme_embeddings[morpheme])
        distances[morpheme] = distance
    sorted_distances = sorted(distances.items(), key=lambda x: x[1])
    expanded_dict.append(seed)
    for i in range(min(args.topn, len(sorted_distances))):
        word, distance = sorted_distances[i]
        expanded_dict.append(word)

# Write expanded dictionary to output file
expanded_dict_filename = f'expanded_dict_kogpt2.txt'
with open(expanded_dict_filename, 'w') as outfile:
    outfile.write('\n'.join(expanded_dict))

# Get top N similar words for each seed word and write to output file
output_filename = f'output_kogpt2.txt'
with open(output_filename, 'w') as outfile:
    for seed in og_seeds:
        outfile.write(f"Top 5 similar words for '{seed}':\n")
        if seed not in expanded_dict:
            continue
        distances = {}
        seed_embedding = get_gpt2_embedding(seed)
        for word in expanded_dict:
            if word == seed:
                continue
            word_embedding = get_gpt2_embedding(word)
            distance = np.linalg.norm(seed_embedding - word_embedding)
            distances[word] = distance
        sorted_distances = sorted(distances.items(), key=lambda x: x[1])
        for i in range(min(5, len(sorted_distances))):
            word, distance = sorted_distances[i]
            outfile.write(f"{word}: {distance}\n")
        outfile.write("\n")
