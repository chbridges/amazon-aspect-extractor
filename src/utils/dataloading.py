from typing import List
import os
import json
import xml.etree.cElementTree as ET
import math

def load_amazon_multilingual(path: str,
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
    if "dev" in os.listdir(path):
        os.rename(os.path.join(path, "dev"), os.path.join(path, "val"))

    for split in os.listdir(path):
        if not os.path.isdir(os.path.join(path, split)):
            continue
        data[split] = []
        for json_file in os.listdir(os.path.join(path, split)):
            if any(map(json_file.__contains__, select_languages)): #check if language is selected
                for line in open(os.path.join(path, split, json_file), "r"):
                    data[split].append(json.loads(line))

    return data

def load_semeval2015(path: str, categories: List[str] = ["laptops", "restaurants", "hotels"]):
    """Load the SemEval 2015 Task 12 Laptop dataset
    Arguments:
    - path: A path to the dataset directory
    - categories: A list of categories to include data from
    choices are: laptops, restaurants, hotels
    for hotels only test data is available

    Return:
    - data: A dictionary with the reviews for each of the 3 splits train/dev/test
    """
    data = {"train":[], "val":[], "test":[]}
    for filename in os.listdir(path):
        if filename.endswith(".xml") and any(map(filename.lower().__contains__, categories)):
            print("Reading file {}...".format(filename))
            tree = ET.parse(os.path.join(path, filename))
            root = tree.getroot()
            split = []
            for review in root:
                review_text = []
                review_opinions = []

                for sentence in review[0]: #review[0] == sentences field
                    review_text.append(sentence[0].text) #sentence[0] == text
                    sentence_opinions = []
                    if len(sentence) == 1:
                        review_opinions.append([])
                        continue
                    for op in sentence[1]: #sentence[1] == opinions
                        category = op.attrib.get("category").split("#")
                        if len(category) == 1:
                            category.append(None)
                        assert len(category) == 2, "category has wrong format: {}".format(category)
                        pos = (op.attrib.get("from"), op.attrib.get("to"))
                        opinion = SemEvalReviewOpinion(op.attrib.get("target"),
                                                       op.attrib.get("polarity"),
                                                       pos,
                                                       category[0],
                                                       category[1])
                        sentence_opinions.append(opinion)
                    review_opinions.append(sentence_opinions)
                split.append(SemEvalReview(review_text, review_opinions))
            if "train" in filename.lower():
                data["train"].extend(split[:math.floor(len(split)*0.9)])
                data["val"].extend(split[math.floor(len(split)*0.9):])
            elif "test" in filename.lower():
                data["test"].extend(split)
    return data



class SemEvalReviewOpinion(object):
    """An opinion contained in a review of the SemEval Dataset.
    Arguments:
    - target: the target aspect term in the sentence, may be None
    - polarity: the polarity of the opinion
    - target_position: the position of the target term in the review sentence,
    is (0,0) if not existant
    - category: the category the opinion is concerning
    - subcategory: the subcategory, may be None

    Attributes:
    Same as arguments except
    - target: "NULL" is converted to python3's None
    - polarity: either 0/0.5/1, denoting negative/neutral/positive
    - target_position: (None, None) is converted to (0, 0)"""
    polarity_dict = {"negative": 0, "neutral": 0.5, "positive": 1}

    def __init__(self, target: str, polarity: str, target_position, category, subcategory):
        if target == "NULL":
            target = None
        self.target = target
        self.polarity = SemEvalReviewOpinion.polarity_dict.get(polarity)
        if self.polarity == None and not (polarity is None):
            raise(Warning("Unknown polarity {}".format(polarity)))
        try:
            if target_position[0].isdigit() and target_position[1].isdigit():
                self.target_position = (int(target_position[0]), int(target_position[1]))
            else:
                raise(ValueError("Target Position not a digit"))
        except:
            self.target_position = (0, 0)
        self.category = category
        self.subcategory = subcategory

    def shift_pos(self, shift):
        if not self.target_position == (0, 0):
            self.target_position = (self.target_position[0] + shift, self.target_position[1] + shift)

    def __str__(self):
        return "Opinion: " + ", ".join([attr+"="+str(self.__dict__[attr]) for attr in self.__dict__.keys()])

    def __eq__(self, other):
        for attr in self.__dict__.keys():
            if self.__dict__[attr] != other.__dict__[attr]:
                return False
        return True

class SemEvalReview(object):
    """A single review of the SemEval Dataset
    Arguments:
    - text: an array of the review text, split up by sentences
    - opinions: a 2d array of the review opinions, split up by sentences

    Attributes:
    - text: the full review text as a string, one sentence per line
    - opinions: a list of opinions, with target_poitions adjusted for the sentence offset"""

    def __init__(self, text: List[str], opinions: List):
        assert len(text) == len(opinions), "Length of opinion and text list not matching: {} - {}".format(len(opinions), len(text))
        self.text = ""
        self.opinions = []
        for line, ops in zip(text, opinions):
            for op in ops:
                op.shift_pos(len(self.text))
                self.opinions.append(op)
            self.text += line + "\n"

    def __str__(self):
        return self.text + "{ " + "\n".join([str(op) for op in self.opinions]) + " }"

    def remove_text(self, span):
        start, end = span
        assert span[0] >= 0, "Invalid span {}".format(span)
        assert span[1] <= len(self.text) + 1, "Span end {} exceeds review length {}".format(span[1], len(self.text))
        self.text = self.text[:start] + self.text[end:]
        for i, op in enumerate(self.opinions):
            op_start, op_end = op.target_position
            if start <= op_start:
                if end >= op_end:
                    #TODO Decide if it might be better to keep opinion and set target position to (0,0)
                    self.opinions.remove(op) #delete opinion if the keyword is removed
                else:
                    op_start = op_start - min(op_start - start, end - start)

            if start <= op_end:
                op_end = op_end - min(end - start, op_end - start) #end and op_end both exclusive
            op.target_position = (op_start, op_end)
