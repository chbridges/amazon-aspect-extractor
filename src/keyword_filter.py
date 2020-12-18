import csv          # Dataset loading
import spacy        # PoS tags
import numpy as np

# Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier

# Evaluation
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_fscore_support

# load dataset

data = []

with open("data/sample_labelled.csv") as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        data.append(row)
data = data[1:]

labels = np.array([bool(int(data[i][1])) for i in range(len(data))])
data = [data[i][0] for i in range(len(data))]

# create PoS tag model

nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

tags = list(spacy.glossary.GLOSSARY.keys())
tags = tags[:tags.index("X")+1]

tagmap = {tags[i]:i for i in range(len(tags))}

# create matrix

matrix = np.zeros((len(data),len(tags)+1))

for i in range(len(data)):
    for word in nlp(data[i]):
        j = tagmap[word.pos_]
        matrix[i,j] += 1
        matrix[i,-1] += 1

# classifier evaluation

def evaluate_classifier(model, splits=5):
    P = []
    R = []
    F1 = []

    kf = KFold(n_splits=splits, shuffle=True, random_state=0)

    for train, test in kf.split(data):
        X_train, X_test = matrix[train], matrix[test]
        y_train, y_test = labels[train], labels[test]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        scores = precision_recall_fscore_support(y_test, y_pred, average='binary')
        P.append(scores[0])
        R.append(scores[1])
        F1.append(scores[2])
    
    print("Precision:", sum(P)/len(P))
    print("Recall:   ", sum(R)/len(R))
    print("F1 score: ", sum(F1)/len(F1))

# default models
print("Logistic Regression:")
evaluate_classifier(LogisticRegression(solver='lbfgs'))
print("\nDecision Tree:")
evaluate_classifier(DecisionTreeClassifier())
print("\nRandom Forest:")
evaluate_classifier(RandomForestClassifier(n_estimators=100))
print("\nGBD:")
evaluate_classifier(GradientBoostingClassifier())
print("\nSVM:")
evaluate_classifier(LinearSVC(tol=1e-3))
print("\nMLP:")
evaluate_classifier(MLPClassifier(solver='lbfgs'))