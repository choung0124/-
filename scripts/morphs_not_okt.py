import codecs
import sys
from gensim.models import Word2Vec
from konlpy.tag import Okt, Kkma, Mecab, Komoran, Hannanum 

# Check if the input file and class name are provided as arguments
if len(sys.argv) != 3:
    print("Usage: python script.py input_file class_name")
    sys.exit(1)

input_file = sys.argv[1]
class_name = sys.argv[2]

# Select the appropriate class based on the input argument
if class_name == "Okt":
    k = Okt()
elif class_name == "KKma":
    k = Kkma()
elif class_name == "Mecab":
    k = Mecab()
elif class_name == "Komoran":
    k = Komoran()
elif class_name == "Hannanum":
    k = Hannanum()
else:
    print("Invalid class name:", class_name)
    sys.exit(1)

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