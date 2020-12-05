import os

from utils.keywords import extract_keyword_list  # import of module from subfolder
from utils.dataloading import load_amazon_mulitlingual

if __name__ == "__main__":
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data") #path to dataset
    dataset = load_amazon_mulitlingual(path, select_languages=["en"])
    reviewtext = [rev["review_body"] for rev in dataset["train"]]
    keywords = extract_keyword_list(reviewtext, minScore=2.0)
    print(keywords[:10])
