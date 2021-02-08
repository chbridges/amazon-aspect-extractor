import csv  # Dataset loading
import os

import numpy as np
import spacy  # PoS tags

import xgboost as xgb


class Filter:
    def __init__(self, params=dict()):
        self.nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
        self.params = params
        self.bst = None

        # Important paths
        self.current_path = os.path.dirname(os.path.abspath(__file__))
        self.data_path = os.path.join(self.current_path, "../data/")

        # Encoded POS tags
        tags = list(spacy.glossary.GLOSSARY.keys())
        self.tags = tags[: tags.index("X") + 1]  # we only want the simple pos_ tags
        self.tagmap = {tags[i]: i for i in range(len(tags))}
        self.tags.append("WORDCOUNT")

    def train(self, cv=False):
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
                matrix[i, -1] += 1

        dtrain = xgb.DMatrix(data=matrix, label=labels, feature_names=self.tags)

        # Training
        print("Training the XGBoost model...")
        bst = xgb.train(params=self.params, dtrain=dtrain)
        bst.save_model(os.path.join(self.current_path, "keywordfilter.model"))

        if cv:
            print(xgb.cv(dict(), dtrain))

        self.bst = bst
        return self.bst

    def load_model(self):
        self.bst = xgb.Booster()
        try:
            self.bst.load_model(os.path.join(self.current_path, "keywordfilter.model"))
        except xgb.core.XGBoostError:
            print("ERROR: There is no pretrained model, call Filter.train() instead.")
            return None
        return self.bst

    def predict(self, keyword: str):
        if self.bst == None:
            print("ERROR: Model is not trained.")
            return None

        vector = np.zeros((1, len(self.tags)))
        for word in self.nlp(keyword):
            idx = self.tagmap[word.pos_]
            vector[0, idx] += 1
            vector[0, -1] += 1

        data = xgb.DMatrix(vector, feature_names=self.tags)

        return self.bst.predict(data)


# Sample usage and some insights for the report
if __name__ == "__main__":
    bst = Filter()

    if bst.load_model() == None:
        bst.train(cv=True)

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
