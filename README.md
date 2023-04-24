# Requirements

``` ! pip install gensim ```
``` ! pip install konlpy ```
``` ! pip install google-api-python-client ```
``` ! pip install sklearn ```

# Models

Models can be downloaded from here http://nlp.kookmin.ac.kr/kcc/word2vec/

Both FastText and Word2Vec models will work with these scripts.

There is an empty directory called models. When downloading the model zip files extract them anywhere you'd like, 
BUT make sure you copy the .model file into the /models directory.

# Scripts

## Run all scripts from the root directory

### Preparing the corpus

#### Crawling youtube comments

1. Find the channel IDs of the youtube channels you want to crawl comments from

``` ! python scripts/channel_id_crawling.py "Youtube channel username" ```

    Replace "Youtube channel username" with the username of your channel, this must be exact

    The channel IDs will be saved to "channel_ids.csv" in your current directory. 
    This will be used as the input for crawling, so review this file before the proceding to the next step

2. Crawl the all of the comments of each youtube channel

``` ! python scripts/yt_channel_crawling.py "channel_ids.csv" ``` 

    This script will crawl all of the comments and combine them into a file in your current directory called "extracted_text_korean_only.txt". 
    This is your corpus. 
    This file is encoded in euc-kr so when opening this file make sure to use an editor like VS Code that can decode euc-kr otherwise the text will look like this: "�Ҿ���� �� �����԰� �ִ� ���� ��Ż������ �� �ϰ� �ִ� �ſ����� �� �"
    At the bottom right of vs code you will see a button called "UTF-8" click on that and select the option to reopen the file and use euc-kr.

### Extracting comments from AI_HUBs JSON files

1. Extract comments from JSON files

``` ! python scripts/crawl_json.py "/Json Directory" ```

    Replace "/Json Directory" with a path to a folder that contains only json files *NO FOLDERS*
    For example: "project/원천데이터/TS1/건강_의학"
    This will output a .txt file named "건강_의학_output.txt" to your current directory
    Make sure that you organize these txt files into their own directory, this will help with the next step. 
    Again, these files are encoded in euc-kr, refer to the previous steps on how to read euc-kr files

2. Combine the output.txt files into your final corpus

``` ! python scripts/combine_yt_comments.py "outputs directory" ```

    Replace "outputs directory" with the directory you made for the outputs from step 1. 
    It is crucial that only the output.txt files are in this directory, otherwise you will get unwanted text in your final corpus.
    This outputs a file called "combined_corpus.txt", encoded in euc-kr, to your current directory

### Morpheme Generation(형태소 분석석)

1. Convert your corpus into a morpheme"형태소" dictionary

``` ! python scripts/morphs.py "your corpus.txt file" ```

    Replace "your corpus.txt file" with the path to your corpus file, which will either be "yourcurrentdir/combined_corpus.txt" or "yourcurrentdir/extracted_text_korean_only.txt" depending on whether you crawled from youtube or AI-HUB.
    This outputs a file whose name begins with your input filename, and ends in morphs.txt
    For example: "combined_corpus_morphs.txt". This is your morph dictionary

###  Dictionary Expansion()

1. Expanding your dictionary, 1st time

``` ! python scripts/dictionary_expansion.py "morph dictionary" "models/yourmodel" "seed_words.txt" number```

    Replace "morph dictionary" with the path to the morph dictionary you created in the previous step
    Replace "model" with the path to the model you'd like to use to measure similarity
    There is no need to replace "seed_words.txt" as it is already in your current directory. Open this file and replace the 5 words with 5 seed words of your own choice.
    Replace number with the number of similar words you'd like to expand by, per seed word.

    Once you run the script, this will create two output files. One will be named expanded_dict_modelname e.g. expanded_dict_FastText-KCC150.model.txt, this is your expanded dictionary
    The second file will be called output_modelname.txt e.g. output_FastText-KCC150.model.txt, this will show you the top5 similar words to your seed words and their similarity score.

    Change the name of bothe the output files before proceeding with the next step.

2. Expanding your dictionary, 2nd time

``` ! python scripts/dictionary_expansion.py "morph dictionary" "models/yourmodel" "seed_words.txt" number```

    This is exactly the same as step 1, but this time replace "seed_words.txt" with the expanded dictionary txt file you got from the previous step, you should have changed the name of this file by now, otherwise this script will delete its contents to write over it.





