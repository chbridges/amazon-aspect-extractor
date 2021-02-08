import csv  # Dataset loading
import os

import numpy as np
import spacy  # PoS tags

import xgboost as xgb


class Filter:
    def __init__(self, include_wordlength=False, params=dict()):
        self.nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
        self.params = params
        self.bst = None
        self.dtrain = None

        # Important paths
        self.current_path = os.path.dirname(os.path.abspath(__file__))
        self.data_path = os.path.join(self.current_path, "../data/")

        # Encoded POS tags
        tags = list(spacy.glossary.GLOSSARY.keys())
        self.tags = tags[: tags.index("X") + 1]  # we only want the simple pos_ tags
        self.tagmap = {tags[i]: i for i in range(len(tags))}
        self.include_wordlength = include_wordlength
        if self.include_wordlength:
            self.tags.append("WORDCOUNT")

    def train(self):
        # Load data
        data = []

        with open(os.path.join(self.data_path, "filter_train.csv")) as csvfile:
            reader = csv.reader(csvfile, delimiter=",")
            for row in reader:
                data.append(row)
        data = data[1:]

        labels = np.array([bool(int(data[i][1])) for i in range(len(data))])
        data = [data[i][0] for i in range(len(data))]

        # Create PoS tag matrix
        matrix = np.zeros((len(data), len(self.tags)))

        for i in range(len(data)):
            for word in self.nlp(data[i]):
                j = self.tagmap[word.pos_]
                matrix[i, j] += 1
                if self.include_wordlength:
                    matrix[i, -1] += 1

        self.dtrain = xgb.DMatrix(data=matrix, label=labels, feature_names=self.tags)
        self.dtrain.save_binary(os.path.join(self.current_path, "keywordfilter.buffer"))

        # Training
        print("Training the XGBoost model...")
        self.bst = xgb.train(params=self.params, dtrain=self.dtrain)
        self.bst.save_model(os.path.join(self.current_path, "keywordfilter.model"))

        return self.bst

    def load_model(self):
        print("Loading XGBoost model...")
        try:
            self.bst = xgb.Booster()
            self.bst.load_model(os.path.join(self.current_path, "keywordfilter.model"))
            self.dtrain = xgb.DMatrix(
                os.path.join(self.current_path, "keywordfilter.buffer")
            )
        except xgb.core.XGBoostError:
            print("ERROR: There is no pretrained model, call Filter.train() instead.")
            return None
        return self.bst

    def predict(self, keywords: list):
        if self.bst is None:
            print("ERROR: Model is not trained.")
            return None

        if type(keywords) == str:
            keywords = [keywords]

        matrix = np.zeros((0, len(self.tags)))

        for kw in keywords:
            vector = np.zeros((1, len(self.tags)))
            for word in self.nlp(kw):
                idx = self.tagmap[word.pos_]
                vector[0, idx] += 1
                if self.include_wordlength:
                    vector[0, -1] += 1
            np.vstack((matrix, vector))

        data = xgb.DMatrix(matrix, feature_names=self.tags)

        return self.bst.predict(data)

    def cv(self):
        if self.dtrain is None:
            print("ERROR: Model is not trained")
            return

        results = xgb.cv(dict(), self.dtrain, nfold=5, as_pandas=False)
        print("\nResults of 5-fold cross-validation:")
        print("train-rmse-mean:", np.mean(results["train-rmse-mean"]))
        print("train-rmse-std: ", np.mean(results["train-rmse-std"]))
        print("test-rmse-mean: ", np.mean(results["test-rmse-mean"]))
        print("test-rmse-std:  ", np.mean(results["test-rmse-std"]), "\n")


# Sample usage and some insights for the report
if __name__ == "__main__":
    bst = Filter(include_wordlength=False)

    if bst.load_model() is None:
        bst.train()

    bst.cv()

    print("good battery life:", bst.predict("good battery life"))
    print("I am amazed:", bst.predict("I am amazed"))
    print("okay:", bst.predict("okay"))
    print("customer support:", bst.predict("customer support"))
    print("lenovo:", bst.predict("lenovo"))
    print(
        "this one is impossible but long:",
        bst.predict("this one is impossible but long"),
    )
    print("service:", bst.predict("service"))

    # Requires matplotlib which is not in Pipfile or requirements.txt

    try:
        from matplotlib import pyplot as plt

        xgb.plot_importance(bst.bst)
        plt.show()

    except ModuleNotFoundError:
        pass
