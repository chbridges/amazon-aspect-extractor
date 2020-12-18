import en_core_web_sm
import spacy
import re
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


def remove_special_characters(review):
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


def remove_stopwords(review):
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


def review_to_int(reviewtexts):
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


def stemming(reviewtext):
    raise (NotImplementedError("Stemming not implemented"))


def tokenize(review):
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
                        remove_special_characters,
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
                        remove_stopwords,
                        tqdm(
                            reviews, desc="Removing stopwords", leave=False, position=1
                        ),
                    )
                )
            if self.tokenize:
                reviews = list(
                    map(
                        tokenize,
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
                        stemming,
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
