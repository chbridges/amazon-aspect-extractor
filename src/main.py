import os

from utils.keywords import extract_keyword_list  # import of module from subfolder
from utils.dataloading import load_amazon_mulitlingual, load_semeval2015

if __name__ == "__main__":
    # path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/amazon_multilingual") #path to dataset
    # dataset = load_amazon_mulitlingual(path, select_languages=["en"])
    # reviewtext = [rev["review_body"] for rev in dataset["train"]]
    # keywords = extract_keyword_list(reviewtext, minScore=2.0)
    # print(keywords[:10])
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/ABSA15")
    dataset = load_semeval2015(path, categories=["hotels"])
    for rev in dataset["test"][:5]:
        print(rev)
