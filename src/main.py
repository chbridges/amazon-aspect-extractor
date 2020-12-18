import os
import torch

from utils.keywords import extract_keywords_list
from utils.dataloading import load_amazon_multilingual, load_semeval2015
from utils.preprocessing import PreprocessingPipeline
from utils.sentiment import (
    SentimentModel,
    SentimentDataset,
    train_sentiment_model,
    evaluate_sentiment_model,
    cross_entropy,
)
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR
from torch.cuda import is_available
from utils.metrics import cross_entropy, f1_score, accuracy, class_balanced_accuracy


if __name__ == "__main__":
    # path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/amazon_multilingual") #path to dataset
    # dataset = load_amazon_mulitlingual(path, select_languages=["en"])
    # reviewtext = [rev["review_body"] for rev in dataset["train"]]
    # keywords = extract_keyword_list(reviewtext, minScore=2.0)
    # print(keywords[:10])
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/ABSA15")
    dataset = load_semeval2015(path, categories=["restaurants"])
    pipeline = PreprocessingPipeline()
    dataloaders = {}

    dataset, dict_for, dict_back = pipeline(dataset)
    for phase in dataset.keys():
        revtexts, aspects, sentiment = dataset[phase]
        sentiments = SentimentDataset(revtexts, aspects, sentiment)
        dataloaders[phase] = DataLoader(sentiments, batch_size=32, shuffle=True)

    device = "cuda" if is_available() else "cpu"
    model = SentimentModel(
        len(dict_for) + 1,
        output_size=3,
        model_name="LSTM_classifier",
        device=device,
        normalize_output=False,
        n_layers=1,
        embedding_dim=128,
        hidden_dim=768,
        dropout=0.4,
        bidirectional=False,
    )
    optimizer = Adam(model.parameters(), lr=0.004, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, cooldown=10, factor=0.2, verbose=True)
    criterion = cross_entropy(dataloaders["train"], device=device)
    # train_sentiment_model(model, optimizer, dataloaders, criterion=criterion, scheduler=scheduler, n_epochs=150, eval_every=1)
    model.load_state_dict(
        torch.load(os.path.join("models", model.name + "_best" + ".pth"))[
            "model_state_dict"
        ]
    )
    for metric in [accuracy, class_balanced_accuracy, f1_score]:
        evaluate_sentiment_model(model, dataloaders, criterion=metric)
