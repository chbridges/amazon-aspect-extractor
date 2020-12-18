import RAKE
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from tqdm import tqdm
from typing import List


def extract_keywords_str(
    document: str, maxWords: int = -1, minScore: float = 0.0, useNLTK: bool = True
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

    if useNLTK:
        stoplist = stopwords.words("english")
    else:
        stoplist = RAKE.SmartStopList()

    rake = RAKE.Rake(stoplist)
    keywords = rake.run(
        document
    )  # dont use maxWords here, in case we filter words with minScore
    keywords = [k for k in keywords if k[1] >= minScore]
    keywords = keywords[:maxWords]  # Sorted by Score
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


def keywords_to_dataframe(keywords: list, csv_name: str = ""):
    """Converts a list of extracted keywords to an appropriate Pandas DataFrame
    Arguments:
    - keywords: A list of extracted keywords (using the above functions)
    - csv_name: If non-empty, the DataFrame will be saved in data/<csv_name>.csv

    Return:
    - df: A dataframe with object column "keyword" and float column "relevancy"
    """
    df = pd.DataFrame(keywords, columns=["keyword", "relevancy"])

    if csv_name != "":
        df.to_csv(f"data/{csv_name}.csv", index=False)

    return df
