import re
import RAKE
import spacy
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List
from nltk.corpus import stopwords


def extract_keywords_str(
    document: str,
    minWords: int = 1,
    maxKeywords: int = -1,
    minScore: float = 0.0,
    useNLTK: bool = True,
) -> List[str]:
    """Extract the top keywords using RAKE
    Arguments:
    - document: A string containing the text to extract from
    - maxWords: An integer indicating how many keywords to extract
    - minScore: A float that filters out any keywods with lower Score than this
    - useNLTK:  A boolean to choose between NLTK's stoplist and RAKE's SmartStoplist

    Return:
    - keywords: A list of the keywords in the document as (keyword, relevancy) tuples
    """
    if type(document) in (list, set, tuple):
        raise TypeError(
            f"Function extract_keyword_str expects string type, got {type(document)} type instead. Did you mean to use extract_keyword_list?"
        )

    def count_words(keyword: str):
        count = 1
        for c in keyword:
            count += c == " "
        return count

    if useNLTK:
        stoplist = stopwords.words("english")
    else:
        stoplist = RAKE.SmartStopList()

    rake = RAKE.Rake(stoplist)
    keywords = rake.run(
        document
    )  # dont use maxWords here, in case we filter words with minScore
    keywords = [
        k for k in keywords if k[1] >= minScore and count_words(k[0]) >= minWords
    ]
    keywords = keywords[:maxKeywords]  # Sorted by Score
    return keywords


def extract_keywords_list(documents: List[str], **kwargs) -> List:
    """Return top keywords using RAKE for a list of documents
    Arguments:
    - document: A list containing the documents to extract from
    - kwargs: Keywords arguments to extract_keyword_str

    Return:
    - keywords: A list of the keywords in the documents (keyword, relevancy) tuples
    """
    if type(documents) == str:
        raise TypeError(
            "Function extract_keyword_list expects list type, got string type. Did you mean to use extract_keyword_str?"
        )

    keywords = []
    for doc in tqdm(documents, "Extracting keywords"):
        keywords.extend(extract_keywords_str(doc, **kwargs))
    keywords.sort(key=lambda x: x[1], reverse=True)  # Sort by descending Score
    return keywords


def keywords_to_dataframe(
    keywords: list, include_relevancy: bool = True, csv_name: str = ""
):
    """Converts a list of extracted keywords to an appropriate Pandas DataFrame
    Arguments:
    - keywords: A list of extracted keywords (using the above functions)
    - include_relevancy: A boolean whether or not to include the relevancy score as a second column or not
    - csv_name: If non-empty, the DataFrame will be saved in data/<csv_name>.csv

    Return:
    - df: A dataframe with object column "keyword" and float column "relevancy"
    """

    if include_relevancy:
        df = pd.DataFrame(keywords, columns=["keyword", "relevancy"])
    else:
        keywords = [x[0] for x in keywords]
        df = pd.DataFrame(keywords, columns=["keyword"])

    if csv_name != "":
        df.to_csv(f"data/{csv_name}.csv", index=False)

    return df


def find_aspects_str(
    doc: str, ignore_adjectives: bool = True, return_as_int: bool = False
) -> str:
    """
    Returns a boolean vector where each entry corresponds to a token in the given review.
    True means that the token belongs to an aspect, False means otherwise.
    Optionally set all adjectives and adverbs to False as they can belong to the sentiment.
    Arguments:
    - document: A string, typically containing a review
    - ignore_adjectives: Boolean, if True all adjectives and adverbs will be set to False.
    - return_as_int: Boolean, if True the list will contain values 0 and 1
    Return:
    List of booleans
    """
    if ignore_adjectives:
        nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
        pos = lambda w: nlp(w)[0].pos_

    # Different results using NLTK stoplist and SmartStoplist, so we combine both
    keywords_nltk = [k[0] for k in extract_keywords_str(doc)]
    keywords_smart = [k[0] for k in extract_keywords_str(doc, useNLTK=False)]
    keywords = list(set(keywords_nltk + keywords_smart))

    doc_tokens = re.split(" |\n", re.sub(r"[^\w\d\s]|'", "", doc).lower())
    if "" in doc_tokens:
        doc_tokens.remove("")

    doc_len = len(doc_tokens)
    aspectvector = [False for i in range(doc_len)]

    for kw in keywords:
        kw_tokens = re.split(" |\n", kw)
        kw_len = len(kw_tokens)
        for i in range(doc_len - kw_len):
            if doc_tokens[i : i + kw_len] == kw_tokens:
                if ignore_adjectives:
                    subvec = [pos(t) not in ("ADJ", "ADV") for t in kw_tokens]
                else:
                    subvec = [True] * kw_len
                aspectvector[i : i + kw_len] = subvec

    if return_as_int:
        return [int(x) for x in aspectvector]

    return aspectvector
