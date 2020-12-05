from typing import List
import os
import json

def load_amazon_mulitlingual(path: str,
                             select_languages: List[str] =
                             ["en", "de", "zh", "es", "fr", "ja"]) -> dict:
    """Load the json formatted aws mulitlingual dataset
    Arguments:
    - path: A path to the dataset directory
    - select_languages: A list of languages to filter the data for

    Return:
    - data: A dictionary with the reviews for each of the 3 splits train/dev/test
    """
    data = {}
    for split in os.listdir(path):
        if not os.path.isdir(os.path.join(path, split)):
            continue
        data[split] = []
        for json_file in os.listdir(os.path.join(path, split)):
            if any(map(json_file.__contains__, select_languages)): #check if language is selected
                for line in open(os.path.join(path, split, json_file), "r"):
                    data[split].append(json.loads(line))

    return data

def load_semeval2015(path: str, categories: List[str] = ["laptops", "restaurants"]):
    raise(NotImplementedError())
