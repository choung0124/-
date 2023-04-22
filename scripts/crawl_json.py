import json
import os
import sys
import re
import codecs

if len(sys.argv) < 2:
    print("Usage: python script_name.py <directory_name>")
    sys.exit(1)

# set the directory path where the JSON files are located
dir_path = sys.argv[1]

# create an empty list to store the content from each file
contents = []

# loop through each file in the directory
for file_name in os.listdir(dir_path):
    if file_name.endswith('.json'):
        # read the JSON file and extract the content field from each object
        with open(os.path.join(dir_path, file_name), 'r') as file:
            data = json.load(file)
            for obj in data['SJML']['text']:
                content = obj['content']
                # split the content into sentences by periods and write each sentence to a separate line in the output file
                for sentence in content.split('.'):
                    sentence = sentence.strip()
                    if sentence:
                        korean_reply = re.findall('[ㄱ-ㅎㅏ-ㅣ가-힣]+', sentence)
                        korean_sentence = ' '.join(korean_reply)
                        if korean_sentence:
                            contents.append(korean_sentence)

# create the output file name
output_file_name = f"{os.path.basename(dir_path)}_output.txt"
print(len(contents))
# write the list of contents to a file
with codecs.open(output_file_name, 'w', encoding='euc-kr') as file:
    for sentence in contents:
        file.write(sentence + '\n')
