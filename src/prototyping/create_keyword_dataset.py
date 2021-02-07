#############################################################
# This module will only work if placed into the src folder! #
#############################################################

from utils.dataloading import load_amazon_multilingual, load_semeval2015
from utils.keywords import (
    extract_keywords_list,
    extract_keywords_str,
    keywords_to_dataframe,
)

# data = load_amazon_multilingual("data/amazon-reviews-ml", ['en'])
data = load_semeval2015("data/ABSA15")

train = data["train"]
val = data["val"]
test = data["test"]

full = train + val + test

sample = [full[i].text for i in range(len(full)) if i % (len(full) // 100) == 0]


def get_keywords(doc):
    keywords_nltk = [k for k in extract_keywords_str(doc)]
    keywords_smart = [k for k in extract_keywords_str(doc, useNLTK=False)]
    return list(set(keywords_nltk + keywords_smart))


keywords = []
for s in sample:
    keywords.extend(get_keywords(s))

df = keywords_to_dataframe(keywords[:200], False, csv_name="sample")
