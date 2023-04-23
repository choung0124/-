import numpy as np
from gensim.models import KeyedVectors, Word2Vec, FastText
import argparse
import os

# Set up argument parser
parser = argparse.ArgumentParser()
parser.add_argument("morpheme_file", help="Path to morpheme txt file")
parser.add_argument("model_file", help="Path to pre-trained word2vec model")
parser.add_argument("seed_file", help="Path to seed words txt file")
parser.add_argument("topn", type=int, default=5, help="Number of most similar words to include in the expanded dictionary")
args = parser.parse_args()

# Load morpheme dictionary
morpheme_dict = {}
with open(args.morpheme_file, encoding="euc-kr") as f:
    for line in f:
        line = line.strip().split()
        morpheme_dict[line[0]] = np.array([float(x) for x in line[1:]])

# Calculate word embeddings for all words in the morpheme dictionary
morpheme_embeddings = {}
for morpheme in morpheme_dict:
    morpheme_embeddings[morpheme] = np.zeros(len(morpheme_dict[morpheme]))
    for subword in morpheme_dict[morpheme]:
        morpheme_embeddings[morpheme] += subword
    morpheme_embeddings[morpheme] /= len(morpheme_dict[morpheme])

# Load pre-trained word embedding model
try:
    model = Word2Vec.load(args.model_file)
    model_vocab = model.wv.vocab
except:
    model = FastText.load(args.model_file)
    model_vocab = model.wv.key_to_index

# Load seed words
with open(args.seed_file) as f:
    seeds = [line.strip() for line in f]

# Get top N similar words for each seed word and write to output file
output_filename = f'output_{os.path.basename(args.model_file)}.txt'
with open(output_filename, 'w') as outfile:
    for seed in seeds:
        outfile.write(f"Top {args.topn} similar words for '{seed}':\n")
        if seed in model_vocab:
            seed_embedding = model.wv[seed]
        else:
            continue
        similarities = {}
        for morpheme in morpheme_embeddings:
            similarity = np.dot(seed_embedding, morpheme_embeddings[morpheme])
            similarities[morpheme] = similarity
        sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        for i in range(args.topn):
            word, similarity = sorted_similarities[i]
            outfile.write(f"{word}: {similarity}\n")
        outfile.write("\n")

# Create expanded dictionary with the N most similar words for each seed word
expanded_dict = []
for seed in seeds:
    if seed in model_vocab:
        seed_embedding = model.wv[seed]
    else:
        continue
    similarities = {}
    for morpheme in morpheme_embeddings:
        similarity = np.dot(seed_embedding, morpheme_embeddings[morpheme])
        similarities[morpheme] = similarity
    sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    expanded_dict.append(seed)
    for i in range(args.topn):
        word, similarity = sorted_similarities[i]
        expanded_dict.append(word)

# Write expanded dictionary to output file
expanded_dict_filename = f'expanded_dict_{os.path.basename(args.model_file)}.txt'
with open(expanded_dict_filename, 'w') as outfile:
    outfile.write(', '.join(expanded_dict))
