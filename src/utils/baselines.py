import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import SVC


class SentimentSVM(SVC):
    def __init__(
        self,
        ngrams,
        kernel="poly",
        mode="classifier",
        reg=1.0,
        balanced=True,
        degree=3,
        ex=1,
    ):
        self.ngrams = ngrams
        self.ex = ex
        if mode == "classifier":
            class_weight = "balanced" if balanced else None
            super().__init__(
                kernel=kernel, C=reg, class_weight=class_weight, degree=degree
            )
        elif mode == "regression":
            raise (NotImplementedError())
        else:
            raise (ValueError("Unknown SVM mode {}".format(mode)))

    def fit(self, x, y):
        X = self.extract_features(x)
        return super().fit(X, y)

    def __call__(self, x):
        X = self.extract_features(x)
        return super().predict(X)

    def extract_features(self, x):
        X = np.zeros((len(x), len(self.ngrams)))
        revs, asps = x[:, :, 0], x[:, :, 1]

        distance_vector = np.zeros(asps.shape)
        indx = (
            np.arange(x.shape[1])
            .repeat(x.shape[0], 0)
            .reshape(x.shape[0], x.shape[1], 1)
        )
        for i, asp in enumerate(asps):
            indx_asp = np.where(asp == 1)[0]
            if len(indx_asp):
                indx_asp = indx_asp.reshape(1, len(indx_asp))
                dist = (
                    np.reciprocal(np.min(np.abs(indx[i] - indx_asp), axis=-1) + 1)
                    ** self.ex
                )
                distance_vector[i] = dist
            else:
                distance_vector[i] = np.ones(distance_vector.shape[1])

        for j, ngram in enumerate(self.ngrams):
            ngram_occurence = np.all(rolling_window(revs, len(ngram)) == ngram, axis=-1)
            ngram_occurence = np.concatenate(
                (ngram_occurence, np.zeros((len(revs), len(ngram) - 1))), axis=-1
            )
            ngram_score = np.sum(ngram_occurence * distance_vector, axis=-1)
            X[:, j] = ngram_score
        return X


class SentimentForest(RandomForestClassifier):
    def __init__(
        self,
        ngrams,
        n_trees=25,
        max_depth=20,
        min_samples_split=4,
        bootstrap=True,
        balanced=True,
        mode="classifier",
        pruning=0.0,
        ex=1,
    ):
        self.ngrams = ngrams
        self.ex = ex
        if mode == "classifier":
            class_weight = "balanced_subsample" if balanced else None
            super().__init__(
                n_estimators=n_trees,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                class_weight=class_weight,
                bootstrap=bootstrap,
                ccp_alpha=pruning,
            )
        elif mode == "regression":
            raise (NotImplementedError())
        else:
            raise (ValueError("Unknown RF mode {}".format(mode)))

    def fit(self, x, y):
        X = self.extract_features(x)
        return super().fit(X, y)

    def __call__(self, x):
        X = self.extract_features(x)
        return super().predict(X)

    def extract_features(self, x):
        X = np.zeros((len(x), len(self.ngrams)))
        revs, asps = x[:, :, 0], x[:, :, 1]

        distance_vector = np.zeros(asps.shape)
        indx = (
            np.arange(x.shape[1])
            .repeat(x.shape[0], 0)
            .reshape(x.shape[0], x.shape[1], 1)
        )
        for i, asp in enumerate(asps):
            indx_asp = np.where(asp == 1)[0]
            if len(indx_asp):
                indx_asp = indx_asp.reshape(1, len(indx_asp))
                dist = (
                    np.reciprocal(np.min(np.abs(indx[i] - indx_asp), axis=-1) + 1)
                    ** self.ex
                )
                distance_vector[i] = dist
            else:
                distance_vector[i] = np.ones(distance_vector.shape[1])

        for j, ngram in enumerate(self.ngrams):
            ngram_occurence = np.all(rolling_window(revs, len(ngram)) == ngram, axis=-1)
            ngram_occurence = np.concatenate(
                (ngram_occurence, np.zeros((len(revs), len(ngram) - 1))), axis=-1
            )
            ngram_score = np.sum(ngram_occurence * distance_vector, axis=-1)
            X[:, j] = ngram_score
        return X


def top_ngrams(review_dataset, number=100, n=2):
    rev_texts = [[], [], []]

    for rev, seq_len, sentiment in review_dataset:
        rev_texts[(2 * sentiment).type(torch.int)].append(
            " ".join([str(i) for i in rev[:, 0].tolist()])
        )

    N_0, N_1, N_2 = len(rev_texts[0]), len(rev_texts[1]), len(rev_texts[2])

    counter = CountVectorizer(stop_words=[0], ngram_range=(1, n), min_df=5)

    S = []

    for polarity in range(len(rev_texts)):
        n_grams = counter.fit_transform(rev_texts[polarity])
        transformer = TfidfTransformer().fit(n_grams)
        df = np.reciprocal(transformer.idf_)
        vocab = np.array(counter.get_feature_names())
        S.append((vocab, df * len(rev_texts[polarity])))

    combined_vocab = []
    [combined_vocab.extend(vocab) for vocab, _ in S]
    combined_vocab = list(set(combined_vocab))

    count_mat = np.zeros((len(S), len(combined_vocab)))

    for s, (vocab, counts) in enumerate(S):
        for i, count in enumerate(counts):
            word = vocab[i]
            index = combined_vocab.index(word)
            count_mat[s, index] = count

    S_10, S_11, S_12 = count_mat[0], count_mat[1], count_mat[2]

    frac0 = np.log((S_10 + 1) / (N_0 + 1)) / np.log(N_0 + 1)
    frac1 = np.log((S_11 + 1) / (N_1 + 1)) / np.log(N_1 + 1)
    frac2 = np.log((S_12 + 1) / (N_2 + 1)) / np.log(N_2 + 1)

    F_0 = np.abs(frac0 - 1 / 2 * (frac1 + frac2))
    F_1 = np.abs(frac1 - 1 / 2 * (frac0 + frac2))
    F_2 = np.abs(frac2 - 1 / 2 * (frac0 + frac1))

    F = F_0 + F_1 + F_2
    top_x = np.argsort(F)[:number]
    return [[int(x) for x in combined_vocab[t].split(" ")] for t in top_x]


def rolling_window(a, size):
    # From https://stackoverflow.com/questions/7100242/python-numpy-first-occurrence-of-subarray
    shape = a.shape[:-1] + (a.shape[-1] - size + 1, size)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
