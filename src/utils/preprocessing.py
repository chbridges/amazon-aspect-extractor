import en_core_web_sm
import spacy
import re
from nltk.stem import PorterStemmer
from tqdm import tqdm
from collections import Counter

nlp = en_core_web_sm.load(disable=["parser", "tagger", "ner"])


def lower(review):
    """Lowercase a review
    Arguments:
    - review: the review of the SemEvalReview class

    Returns:
    The review with lowercased text
    """
    review.text = review.text.lower()
    return review


def remove_special_characters_str(doc: str) -> str:
    """Remove special characters
    Arguments:
    - doc: the string
    Returns:
    The string without special characters
    """
    return re.sub(r"[^\w\d\s]|'", "", doc)


def remove_special_characters_review(review):
    """Remove special characters
    Arguments:
    - review: the review of the SemEvalReview class

    Returns:
    The review without special characters
    """
    pattern = re.compile("[^\w\d\s]|'")

    while True:
        match = pattern.search(review.text)
        if match is None:
            break
        review.remove_text(match.span())
    return review


def remove_stopwords_review(review):
    """Remove Stopwords
    Arguments:
    - review: the review of the SemEvalReview class

    Returns:
    The review without stopwords
    """
    # Could be more efficient if everything was converted to integer first
    stopwords = spacy.lang.en.stop_words.STOP_WORDS
    pattern = re.compile(r"\b(" + r"|".join(list(stopwords)) + r")\b\s*")

    while True:
        match = pattern.search(review.text)
        if match is None:
            break
        review.remove_text(match.span())
    return review


def remove_stopwords_list(tokens: list, aspectvector: list = None) -> list:
    """Remove Stopwords from a list of tokens and optionally the indices from a corresponding boolean vector
    Arguments:
    - tokens: a list of tokens
    - aspectvector: a boolean vector returned when extracting the aspects from the same review

    Returns:
    - The list of tokens without stop words
    - Optionally: The filtered aspect vector of same length as the returned list
    """
    stopwords = spacy.lang.en.stop_words.STOP_WORDS

    if aspectvector == None:
        return [t for t in tokens if t not in stopwords]
    else:
        new_tokens = []
        new_aspectvector = []
        for i in range(len(tokens)):
            if tokens[i] not in stopwords:
                new_tokens.append(tokens[i])
                new_aspectvector.append(aspectvector[i])
        return new_tokens, new_aspectvector


def review_to_int(reviewtexts: list):
    """Create the dictionaries for converting words to numbers and back
    Arguments:
    - reviews: a 2d list of shape (num_reviews, num_tokens), where reviews are saved as strings"""
    reviewtexts = [
        token
        for rev in tqdm(
            reviewtexts, desc="Creating integer token dictionary", leave=False
        )
        for token in rev
    ]
    reviewtexts = Counter(reviewtexts)
    sort = reviewtexts.most_common(len(reviewtexts))

    str_to_int_dict = {w: i + 1 for i, (w, c) in enumerate(sort)}
    int_to_str_dict = {v: k for k, v in str_to_int_dict.items()}
    return str_to_int_dict, int_to_str_dict


def stem_str(doc: str) -> str:
    """Stems all words in a given string
    Arguments:
    - doc: A string (typically review.text)
    Returns:
    - modified doc
    """
    wordpattern = r"\b\w+\b"
    stemmer = PorterStemmer()

    last_index = 0

    while True:
        new_word = re.search(wordpattern, doc[last_index:])
        print(new_word)
        if new_word == None:
            break
        else:
            begin = new_word.span()[0] + last_index
            end = new_word.span()[1] + last_index
            stem = stemmer.stem(doc[begin:end])
            doc = doc[:begin] + stem + doc[end:]
            last_index = begin + len(stem)
    return doc


def stem_list(tokens: list) -> list:
    """Stems all tokens in a given list
    Arguments:
    - tokens: List of tokens

    Returns:
    List of stemmed tokens
    """
    stem = PorterStemmer().stem
    return [stem(t) for t in tokens]


def tokenize_str(doc: str) -> list:
    pass


def tokenize_review(review):
    """Tokenize a review
    Arguments:
    - review: the untokenized review of the SemEvalReview class

    Returns:
    a 2-tuple of lists with review tokens and opinions
    """
    text = re.split(" |\n", review.text)
    text.remove("")
    start = 0
    opinions_tokenized = [[] for op in review.opinions]
    opinion_sentiments = [op.polarity for op in review.opinions]
    for token in text:
        end = start + len(token) + 1
        for i, op in enumerate(review.opinions):
            if (
                op.target_position[0] <= start < op.target_position[1]
                or op.target_position[0] < end <= op.target_position[1]
            ):
                opinions_tokenized[i].append(1)
            else:
                opinions_tokenized[i].append(0)
        start = end
    return text, opinions_tokenized, opinion_sentiments


class PreprocessingPipeline(object):
    """Add multiple preprocessing steps into a pipeline
    Arguments:
    - lower: Whether to lower case the review text
    - rm_special_char: Whether to remove special characters
    - rm_stopwords: Whether to remove stopwords
    - tokenize: Whether to tokenize the text
    If this option is set, applying the pipeline will return a 2-tuple of nested
    lists with tokens, opinions and polarity, instead of the original reviews in
    the dataset
    - stemming: Whether to perform stemming. Needs tokenization
    - rev_to_int: Whether to encode the review text as integers. Needs tokenization
    If this option is set, applying the pipeline will return a 3-tuple of
    transformed review text, opinions, sentiment in the dataset and the forward and
    reverse dict for integer conversion"""

    def __init__(
        self,
        lower=True,
        rm_special_char=True,
        rm_stopwords=True,
        tokenize=True,
        stemming=False,
        rev_to_int=True,
    ):
        if rev_to_int or stemming:
            assert (
                tokenize
            ), "Need to tokenize before stemming or converting to int, but tokenize is set to False"
        self.lower = lower
        self.rm_special_char = rm_special_char
        self.rm_stopwords = rm_stopwords
        self.tokenize = tokenize
        self.stemming = stemming
        self.rev_to_int = rev_to_int

    def __call__(self, dataset):
        """Apply the pipeline
        Arguments:
        - dataset: The dataset dictionary with a list of reviews for each split"""
        for phase in tqdm(
            dataset.keys(), desc="Processing dataset splits", leave=False, position=0
        ):
            reviews = dataset[phase]
            if self.lower:
                reviews = list(
                    map(
                        lower,
                        tqdm(
                            reviews,
                            desc="Converting to lowercase",
                            leave=False,
                            position=1,
                        ),
                    )
                )
            if self.rm_special_char:
                reviews = list(
                    map(
                        remove_special_characters_review,
                        tqdm(
                            reviews,
                            desc="Removing special characters",
                            leave=False,
                            position=1,
                        ),
                    )
                )
            if self.rm_stopwords:
                reviews = list(
                    map(
                        remove_stopwords_review,
                        tqdm(
                            reviews, desc="Removing stopwords", leave=False, position=1
                        ),
                    )
                )
            if self.tokenize:
                reviews = list(
                    map(
                        tokenize_review,
                        tqdm(reviews, desc="Tokenizing", leave=False, position=1),
                    )
                )
                reviews = (
                    [rev[0] for rev in reviews],
                    [rev[1] for rev in reviews],
                    [rev[2] for rev in reviews],
                )
            if self.stemming:
                reviews[0] = list(
                    map(
                        stem_str,
                        tqdm(reviews[0], desc="Stemming", leave=False, position=1),
                    )
                )
            dataset[phase] = reviews

        if self.rev_to_int:
            combined_reviews = [
                rev for phase in dataset.keys() for rev in dataset[phase][0]
            ]
            str_to_int_dict, int_to_str_dict = review_to_int(combined_reviews)
            for phase in tqdm(
                dataset.keys(),
                desc="Processing dataset splits",
                leave=False,
                position=0,
            ):
                reviews = dataset[phase]
                reviews = (
                    [
                        [str_to_int_dict[token] for token in rev]
                        for rev in tqdm(
                            reviews[0],
                            desc="Converting reviews to integer",
                            leave=False,
                            position=1,
                        )
                    ],
                    reviews[1],
                    reviews[2],
                )
                dataset[phase] = reviews
            return dataset, str_to_int_dict, int_to_str_dict
        return dataset
