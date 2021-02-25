import re
from typing import List

import nltk
import pandas as pd
import RAKE
import spacy
from tqdm import tqdm

import yake

nltk.download("stopwords")


def rake_str(
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
    - useNLTK:  A boolean to choose between NLTK stoplist and SmartStoplist

    Return:
    - keywords: A list of (keyword, relevancy) tuples
    """
    if type(document) in (list, set, tuple):
        raise TypeError(
            "Function extract_keyword_str expects string type, "
            f"got {type(document)} type instead. "
            "Did you mean to use rake_list?"
        )

    def count_words(keyword: str):
        count = 1
        for c in keyword:
            count += c == " "
        return count

    if useNLTK:
        stoplist = nltk.corpus.stopwords.words("english")
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


def yake_str(document: str, maxScore: float = 1.0, **kwargs) -> List[str]:
    """Extract the top keywords using YAKE
    Arguments:
    - document: A string containing the text to extract from
    - maxScore: A float that filters out any keywods with greater score than this
    - kwargs:   Arguments for the YAKE keyword extractor
                max_ngram_size = 3
                deduplication_threshold = 0.9
                numOfKeywords = 20

    Return:
    - keywords: A list of (keyword, score) tuples
    """
    if type(document) in (list, set, tuple):
        raise TypeError(
            "Function extract_keyword_str expects string type, "
            f"got {type(document)} type instead. "
            "Did you mean to use yake_list?"
        )

    if document == "":
        return []

    extractor = yake.KeywordExtractor(**kwargs)

    keywords = extractor.extract_keywords(document)
    keywords = [k for k in keywords if k[1] <= maxScore]

    return keywords


def extract_keywords_from_list(
    documents: List[str], algorithm="rake", **kwargs
) -> List:
    """Return top keywords using RAKE or YAKE for a list of documents
    Arguments:
    - documents: A list containing the documents to extract from
    - algorithm: Can be either 'rake' or 'yake'
    - kwargs:    Keywords arguments for the respective extractor

    Return:
    - keywords: A list of (keyword, relevancy) tuples
    """
    if type(documents) == str:
        raise TypeError(
            "Function extract_keywords_from_list expects list type, "
            "got string type instead."
        )

    keywords = []
    if algorithm == "rake":
        extractor = rake_str
        _reverse = True
    elif algorithm == "yake":
        extractor = yake_str
        _reverse = False
    else:
        raise ValueError(
            f"Function extract_keywords_from_list expects algorithm 'rake' or 'yake',"
            "got argument {algorithm} instead."
        )

    for doc in tqdm(documents, "Extracting keywords"):
        keywords.extend(extractor(doc, **kwargs))
    keywords.sort(key=lambda x: x[1], reverse=_reverse)  # Sort by score
    return keywords


def keywords_to_dataframe(
    keywords: list, include_relevancy: bool = True, csv_name: str = ""
):
    """Converts a list of extracted keywords to an appropriate Pandas DataFrame
    Arguments:
    - keywords: A list of extracted keywords (using the above functions)
    - include_relevancy: A boolean to add relevancy score as a second column
    - csv_name: If non-empty, the DF will be saved in data/<csv_name>.csv

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


def aggregate_similar_keywords(kw1: str, kw2: str):
    """Maps 2 keywords to one entity if the are similar.
    Example:
    {battery life, battery} -> battery
    {graphic, graphics} -> graphics
    {graphic quality, graphics} PASS!

    1st run: Unigrams: Base algorithm on edit distance; map to longer element (takes plural forms into account)
    2nd run: Bigrams:  Replace unigrams, map to common unigram?
    """
    pass
