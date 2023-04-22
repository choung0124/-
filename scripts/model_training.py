# ! pip install konlp
# ! pip install gensim
# ! pip install re

from konlpy.tag import Kkma 
from konlpy.utils import pprint 
from konlp.kma.klt2000 import klt2000
import unicodedata
import re

import gensim
from gensim.models import Word2Vec

k = klt2000()

def utf2euc(str):
    return unicodedata(str, 'utf-8').encode('euc-kr')

with open("extracted_text_korean_only.txt", "r", encoding="utf-8") as file:
    morphs_data = []
    for line in file:
        morphs_data.append(k.morphs(utf2euc(line.strip())))

print(morphs_data)

model = Word2Vec(sentences=morphs_data, vector_size=100, window=5, min_count=1, workers=4, sg=1)
model.save("word2vec_model")

w2v_dict = {word: model.wv.get_vector(word) for word in model.wv.key_to_index.keys()}
for key, value in w2v_dict.items():
    # print(key)
    with open ("w2v_dict.txt", 'a', encoding="utf-8") as file:
        file.write(key + "\n")
