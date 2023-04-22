from konlpy.tag import Okt
import codecs
from gensim.models import Word2Vec
import sys

k = Okt()

# Check if the input file is provided as an argument
if len(sys.argv) != 2:
    print("Usage: python script.py input_file")
    sys.exit(1)

input_file = sys.argv[1]

# Create the output file name
output_file = input_file.rsplit(".", 1)[0] + "_morphs.txt"

with codecs.open(input_file, "r", "euc-kr") as f_in:
    morphs_data = []
    for line in f_in:
        # Extract morphs and add them to the list
        morphs = k.morphs(line.strip())
        morphs_data.append(morphs)
        print(morphs)

print("Number of morphs:", sum(len(morphs) for morphs in morphs_data))

with codecs.open(output_file, "w", "euc-kr") as f_out:
    # Write each morph as a separate line in the output file
    for morphs in morphs_data:
        for morph in morphs:
            f_out.write(morph + "\n")

print("Done! Morphs saved to", output_file)
