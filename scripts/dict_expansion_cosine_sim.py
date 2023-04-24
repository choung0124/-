import numpy as np
from gensim.models import KeyedVectors, Word2Vec, FastText
from sklearn.metrics.pairwise import cosine_similarity
import argparse
import codecs
import os

# Set up argument parser
parser = argparse.ArgumentParser()
parser.add_argument("morpheme_file", help="Path to morpheme txt file")
parser.add_argument("model_file", help="Path to pre-trained word2vec model")
parser.add_argument("seed_file", help="Path to seed words txt file")
parser.add_argument("topn", type=int, default=5, help="Number of most similar words to include in the expanded dictionary")
parser.add_argument("--new_seed_file", default=None, help="Path to the new seed_file")
args = parser.parse_args()

# Load pre-trained word embedding model
try:
    model = Word2Vec.load(args.model_file)
    model_vocab = model.wv.vocab
except:
    model = FastText.load(args.model_file)
    model_vocab = model.wv.key_to_index

# Load morpheme dictionary and calculate word embeddings for all morphemes
morpheme_dict = {}
morpheme_embeddings = {}
with codecs.open(args.morpheme_file, encoding="euc-kr") as f:
    for line in f:
        line = line.strip().split()
        morpheme = line[0]
        embeddings = []
        for word in morpheme.split("+"):
            if word in model_vocab:
                embeddings.append(model.wv[word])
        if len(embeddings) > 0:
            morpheme_embeddings[morpheme] = np.mean(embeddings, axis=0)

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
    if seed in model_vocab:
        seed_embedding = model.wv[seed]
    else:
        continue
    similarities = {}
    for morpheme in morpheme_embeddings:
        similarity = cosine_similarity([seed_embedding], [morpheme_embeddings[morpheme]])
        similarities[morpheme] = similarity[0][0]
    sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    expanded_dict.append(seed)
    for i in range(min(args.topn, len(sorted_similarities))):
        word, similarity = sorted_similarities[i]
        expanded_dict.append(word)

# Write expanded dictionary to output file
expanded_dict_filename = f'expanded_dict_{os.path.basename(args.model_file)}.txt'
with open(expanded_dict_filename, 'w') as outfile:
    outfile.write('\n'.join(expanded_dict))

# Get top N similar words for each seed word and write to output file
output_filename = f'output_{os.path.basename(args.model_file)}.txt'
with open(output_filename, 'w') as outfile:
    for seed in og_seeds:
        outfile.write(f"Top 5 similar words for '{seed}':\n")
        if seed not in expanded_dict:
            continue
        similarities = {}
        seed_embedding = model.wv[seed]
        for word in expanded_dict:
            if word == seed:
                continue
            if word in model_vocab:
                word_embedding = model.wv[word]
                similarity = cosine_similarity([seed_embedding], [word_embedding])
                similarities[word] = similarity[0][0]
        sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        for i in range(min(5, len(sorted_similarities))):
            word, similarity = sorted_similarities[i]
            outfile.write(f"{word}: {similarity}\n")
        outfile.write("\n")
