import os
import sys

# Check if the input directory is provided as an argument
if len(sys.argv) != 2:
    print("Usage: python script.py input_directory")
    sys.exit(1)

input_dir = sys.argv[1]

# Get a list of all .txt files in the input directory
txt_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and f.endswith(".txt")]

# Combine the contents of all .txt files into a single list
combined_list = []
for txt_file in txt_files:
    with open(os.path.join(input_dir, txt_file), "r") as f:
        contents = f.read().split(",")
        combined_list.extend(contents)

print("Number of items in combined list:", len(combined_list))

# Write the combined list to a file
output_file = "extracted_text_korean_only.txt"
with open(output_file, "w") as f:
    f.write(",".join(combined_list))

# Merge all sublists into a single list
merged_list = []
for sublist in combined_list:
    merged_list.extend(sublist.split())

print("Number of items in merged list:", len(merged_list))

# Write the merged list to a file
output_file = "merged_corpus.txt"
with open(output_file, "w") as f:
    f.write("\n".join(merged_list))
