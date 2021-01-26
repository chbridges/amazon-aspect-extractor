import re

import spacy

from keywords import rake_str, yake_str


def create_aspectmask(
    doc: str,
    ignore_adjectives: bool = True,
    return_as_int: bool = True,
    algorithm="rake",
) -> str:
    """
    Returns a boolean vector where each entry corresponds to a token.
    True means that the token belongs to an aspect, False means otherwise.
    Arguments:
    - document: A string, typically containing a review
    - ignore_adjectives: Boolean, if True all adjectives and adverbs are False
    - return_as_int: Boolean, if True the list will contain values 0 and 1
    Return:
    List of booleans
    """
    if ignore_adjectives:
        nlp = spacy.load("en", disable=["ner", "parser"])

        def pos(w):
            return nlp(w)[0].pos_

    if algorithm == "rake":
        # Different results for NLTK stoplist and SmartStoplist, so we combine both
        keywords_nltk = [k[0] for k in rake_str(doc)]
        keywords_smart = [k[0] for k in rake_str(doc, useNLTK=False)]
        keywords = list(set(keywords_nltk + keywords_smart))
    elif algorithm == "yake":
        keywords = [k[0] for k in yake_str(doc)]

    # Tokenize doc
    doc_tokens = re.split(" |\n", re.sub(r"[^\w\d\s]|'", "", doc).lower())
    while "" in doc_tokens:
        doc_tokens.remove("")

    doc_len = len(doc_tokens)
    aspectmask = [False for i in range(doc_len)]

    for kw in keywords:
        kw_tokens = re.split(" |\n", kw)  # tokenize keyword
        kw_len = len(kw_tokens)
        for i in range(doc_len - kw_len):
            if (
                doc_tokens[i : i + kw_len] == kw_tokens
            ):  # check for equal token sequences
                if ignore_adjectives:
                    submask = [pos(t) not in ("ADJ", "ADV") for t in kw_tokens]
                else:
                    submask = [True] * kw_len
                aspectmask[i : i + kw_len] = submask

    if return_as_int:
        return [int(x) for x in aspectmask]

    return aspectmask


def count_aspects(mask: list):
    """Counts the number of keywords in a given aspect mask"""
    mask_str = "".join([str(x) for x in mask])
    return len(re.findall(r"^1|01", mask_str))


def split_aspect_mask(input_mask: list):
    """Splits masks containing more than one aspect into multiple masks by "nullifying" distinct aspects
    Example:
    input: [1,1,0,0,1,0]
    output: [[1,1,0,0,0,0], [0,0,0,0,1,0]]
    """
    mask_str = "".join([str(x) for x in input_mask])

    begin_indices = [
        match.start() + (match.start() > 0) for match in re.finditer(r"^1|01", mask_str)
    ]
    end_indices = [
        match.start() + (match.start() > 0) for match in re.finditer(r"10|1$", mask_str)
    ]

    new_masks = [[] for i in range(len(begin_indices))]

    for i in range(len(new_masks)):
        new_masks[i] = [0] * len(input_mask)
        for j in range(begin_indices[i], end_indices[i]):
            new_masks[i][j] = 1

    return new_masks


def extract_aspects(doc: str, sep=r"\."):
    """
    Returns tokenized substrings and corresponding aspect masks for each aspect in a document
    based on a given seperator
    """
    mask = create_aspectmask(doc)
    subdocs = re.split(sep, doc)
    tokenized_subdocs = []
    subsequence_lengths = []
    submasks = []

    # Create token lists of substrings based on the given seperator
    for subdoc in subdocs:
        subdoc_tokens = re.split(" |\n", re.sub(r"[^\w\d\s]|'", "", subdoc).lower())
        while "" in subdoc_tokens:
            subdoc_tokens.remove("")
        tokenized_subdocs.append(subdoc_tokens)
        subsequence_lengths.append(len(subdoc_tokens))

    # Split aspect mask of the doc accordingly
    for i in range(len(subsequence_lengths)):
        begin = sum(subsequence_lengths[:i])
        end = sum(subsequence_lengths[: i + 1])
        submasks.append(mask[begin:end])
        assert len(tokenized_subdocs[i]) == len(
            submasks[i]
        )  # sanity check for equal sizes

    # Filter out subsequences containing no aspects
    #  and split subsequences containing multiple aspects
    filtered_tokens = []
    filtered_masks = []

    for i in range(len(subdocs)):
        if 1 in submasks[i]:  # ignore all masks not containing an extracted keyword
            aspect_count = count_aspects(submasks[i])
            filtered_tokens.extend([tokenized_subdocs[i]] * aspect_count)
            if aspect_count == 1:
                filtered_masks.append(submasks[i])
            else:
                filtered_masks.extend(split_aspect_mask(submasks[i]))

    assert len(filtered_tokens) == len(filtered_masks)  # sanity check for equal sizes

    return filtered_tokens, filtered_masks
