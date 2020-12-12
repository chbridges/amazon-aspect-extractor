import RAKE
from nltk.corpus import stopwords
from tqdm import tqdm
from typing import List


def extract_keyword_str(
    document: str, maxWords: int = -1, minScore: float = 0.0, useNLTK: bool = True
) -> List[str]:
    """Extract the top keywords using RAKE
    Arguments:
    - document: A string containing the text to extract from
    - maxWords: An integer indicating how many keywords to extract
    - minScore: A float that filters out any keywods with lower Score than this
    - useNLTK:  A boolean to choose between NLTK's stoplist and RAKE's SmartStoplist

    Return:
    - keywords: A list of the keywords in the document
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


def extract_keyword_list(documents: List[str], **kwargs) -> List:
    """Return top keywords using RAKE for a list of documents
    Arguments:
    - document: A list containing the documents to extract from
    - kwargs: Keywords arguments to extract_keyword_str

    Return:
    - keywords: A list of the keywords in the document
    """
    if type(documents) == str:
        raise TypeError(
            "Function extract_keyword_list expects list type, got string type. Did you mean to use extract_keyword_str?"
        )

    keywords = []
    for doc in tqdm(documents, "Extracting keywords"):
        keywords.extend(extract_keyword_str(doc, **kwargs))
    keywords.sort(key=lambda x: x[1], reverse=True)  # Sort by descending Score
    return keywords
