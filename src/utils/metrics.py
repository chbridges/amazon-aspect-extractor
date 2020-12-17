import torch
from torch import nn

def log_sq_diff(pred, y):
    """Logarithmic squared difference loss.
    This ranges from 0 to infinity and is subject to minimization
    Arguments:
    - pred: a vector of sentiment predictions between 0 and 1
    - y: a vector of sentiment ground truths between 0 and 1

    Returns:
    logarithmic squared difference of the two vectors"""
    return torch.mean(- torch.log(1 - (pred - y)**2))

def accuracy(pred, y, regression=True, n_classes=3):
    """Accuracy of a prediction pred vs. ground truth y
    Arguments:
    - pred: a vector of sentiment predictions between 0 and 1
    - y: a vector of sentiment ground truths between 0 and 1
    - regression: whether the model output is a regression of the sentiment or the class scores
    - n_classes: The number of classes in the model output

    Returns:
    Accuracy of the predictions when rounded to the nearest label (0/0.5/1)"""
    if regression:
        labels = torch.round(pred*(n_classes-1)).type(torch.int)
    else:
        labels = torch.argmax(pred, dim=-1).type(torch.int)
    labels = labels.squeeze()
    y = (y*(n_classes-1)).type(torch.int)
    return torch.mean((labels == y).float())

def class_balanced_accuracy(pred, y, regression=True, n_classes=3):
    """Mean accuracy per class of a prediction pred vs. ground truth y
    Arguments:
    - pred: a vector of sentiment predictions between 0 and 1
    - y: a vector of sentiment ground truths between 0 and 1
    - regression: whether the model output is a regression of the sentiment or the class scores
    - n_classes: The number of classes in the model output

    Returns:
    Balanced accuracy of the predictions when rounded to the nearest label (0/0.5/1)"""
    if regression:
        labels = torch.round(pred*(n_classes-1)).type(torch.int)
    else:
        labels = torch.argmax(pred, dim=-1).type(torch.int)
    labels = labels.squeeze()
    y = (y*(n_classes-1)).type(torch.int)
    acc = []
    for i in range(n_classes):
        count = torch.sum(y == i)
        if count == 0:
            continue
        tp = torch.sum(torch.where(torch.logical_and(y == i, labels == i), torch.ones(len(y)).to(y.device), torch.zeros(len(y)).to(y.device)))
        acc.append(tp/count)
    return torch.mean(torch.Tensor(acc))

def f1_score(pred, y, regression=True, n_classes=3):
    """F1 Score of a prediction pred vs. ground truth y
    Arguments:
    - pred: a vector of sentiment predictions between 0 and 1
    - y: a vector of sentiment ground truths between 0 and 1
    - regression: whether the model output is a regression of the sentiment or the class scores
    - n_classes: The number of classes in the model output

    Returns:
    F1 Score of the predictions when rounded to the nearest label (0/0.5/1)"""
    if regression:
        labels = torch.round(pred*(n_classes-1)).type(torch.int)
    else:
        labels = torch.argmax(pred, dim=-1).type(torch.int)
    labels = labels.squeeze()
    y = (y*(n_classes-1)).type(torch.int)
    f1_scores = []
    counts = []
    for i in range(n_classes):
        counts.append(torch.sum(y == i))
        if counts[i] == 0:
            f1_scores.append(0) # Doesn't matter what gets appended here
            continue
        tp = torch.sum(torch.where(torch.logical_and(y == i, labels == i), torch.ones(len(y)).to(y.device), torch.zeros(len(y)).to(y.device)))
        fp = torch.sum(torch.where(torch.logical_and(y != i, labels == i), torch.ones(len(y)).to(y.device), torch.zeros(len(y)).to(y.device)))
        fn = torch.sum(torch.where(torch.logical_and(y == i, labels != i), torch.ones(len(y)).to(y.device), torch.zeros(len(y)).to(y.device)))
        f1_scores.append(tp/(tp + 0.5*(fp + fn)))
    counts = torch.Tensor(counts).to(y.device)/len(y)
    return torch.sum(torch.Tensor(f1_scores).to(y.device)*counts)

class cross_entropy(nn.CrossEntropyLoss):

    def __init__(self, trainset, n_classes=3, device="cpu"):

        self.n_classes = n_classes
        sentiments = torch.cat([batch[2] for batch in trainset], dim=-1)
        sentiments = torch.round(sentiments*(n_classes-1))
        counts = []
        for i in range(n_classes):
            counts.append(torch.sum(sentiments == i))
        weights = torch.reciprocal(torch.Tensor(counts)).to(device)
        super().__init__(weights)

    def __call__(self, pred, y):
        y = (y*(self.n_classes -1)).type(torch.long)
        return super().__call__(pred, y)
