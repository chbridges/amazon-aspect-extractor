import os

import numpy as np
import torch
from torch.cuda import is_available
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from utils.baselines import SentimentForest, SentimentSVM, top_ngrams
from utils.dataloading import load_custom_dataset, load_semeval2015
from utils.metrics import (
    accuracy,
    class_balanced_accuracy,
    class_ratio,
    cross_entropy,
    f1_score,
)
from utils.preprocessing import PreprocessingPipeline
from utils.reviewextractor import extract_reviews_for_products
from utils.sentiment import (
    SentimentDataset,
    SentimentModel,
    evaluate_sentiment_model,
    train_sentiment_model,
)

if __name__ == "__main__":
    # path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
    #                       "data/amazon_multilingual") #path to dataset
    # dataset = load_amazon_mulitlingual(path, select_languages=["en"])
    # reviewtext = [rev["review_body"] for rev in dataset["train"]]
    # keywords = extract_keyword_list(reviewtext, minScore=2.0)
    # print(keywords[:10])

    # TODO: Implement Input infrastructure for URLs (-> GUI?)

    # input_url = ['https://www.amazon.com/-/de/dp/B07RF1XD36/ref=lp_16225009011_1_6',
    #             'https://www.amazon.com/dp/B08JQKMFFB/ref=sspa_dk_detail_2?psc=1&pd_rd_i=B08JQKMFFB&pd_rd_w=5AdCg' +
    #             '&pf_rd_p=45e679f6-d55f-4626-99ea-f1ec7720af94&pd_rd_wg=bWbE5&pf_rd_r=HJV72D1QHGE2XJ8QJBV0&pd_rd_r' +
    #             '=b3a4c265-2d13-454f-a385-3ad0a71737eb&spLa=ZW5jcnlwdGVkUXVhbGlmaWVyPUEzN1NSWjVRTFFINUFNJmVuY3J5cHR' +
    #             'lZElkPUEwMjY3OTk2MUQ5ODYwVU4zNlhBVCZlbmNyeXB0ZWRBZElkPUEwMzMwMjc2M1VQMVJXVllMVVpGJndpZGdldE5hbWU9c' +
    #             '3BfZGV0YWlsJmFjdGlvbj1jbGlja1JlZGlyZWN0JmRvTm90TG9nQ2xpY2s9dHJ1ZQ==',
    #             'https://www.amazon.com/product-reviews/B08KH53NKR/ref=cm_cr_arp_d_viewopt_sr?ie=UTF8&filterByStar' +
    #             '=all_stars&reviewerType=all_reviews&pageNumber=1#reviews-filter-bar'
    #             ]
    # crawl list of reviews from input amazon URLs
    # dataset = extract_reviews_for_products(input_url)

    # path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/ABSA15")
    # dataset = load_semeval2015(path, categories=["restaurants"])
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/custom")
    dataset = load_custom_dataset(path, categories=["laptops"])

    pipeline = PreprocessingPipeline()
    dataloaders = {}

    dataset, dict_for, dict_back = pipeline(dataset)
    # ----------------------------------------------------------------------------------------
    # Baseline Models
    #
    revtexts, aspects, sentiment = dataset["train"]
    train_set = SentimentDataset(revtexts, aspects, sentiment)

    revtexts, aspects, sentiment = dataset["val"]
    val_set = SentimentDataset(revtexts, aspects, sentiment)
    top_x = top_ngrams(train_set)
    forest = SentimentForest(top_x, pruning=0.01, ex=2)
    svm = SentimentSVM(top_x, degree=3, ex=2)

    x_train = torch.stack([event for event, _, _ in train_set], axis=0).numpy()
    y_train = torch.Tensor([sentiment for _, _, sentiment in train_set]).numpy()
    y_train = (2 * y_train).astype(np.int)

    x_val = torch.stack([event for event, _, _ in val_set], axis=0).numpy()
    y_val = torch.Tensor([sentiment for _, _, sentiment in val_set]).numpy()
    y_val = (2 * y_val).astype(np.int)
    #print([dict_back[i[0]] for i in top_x])
    print("Fitting SVM...")
    svm.fit(x_train, y_train)
    print("Fitting Random Forest...")
    forest.fit(x_train, y_train)

    y_svm_train = svm(x_train)
    y_svm_val = svm(x_val)
    y_forest_train = forest(x_train)
    y_forest_val = forest(x_val)

    print(
        "SVM train accuracy: {}".format(
            np.mean(
                np.where((y_train).astype(np.int) == y_svm_train.astype(np.int), 1, 0)
            )
        )
    )
    print(
        "SVM val accuracy: {}".format(
            np.mean(np.where((y_val).astype(np.int) == y_svm_val.astype(np.int), 1, 0))
        )
    )
    print(
        "Random Forest train accuracy: {}".format(
            np.mean(
                np.where(
                    (y_train).astype(np.int) == y_forest_train.astype(np.int), 1, 0
                )
            )
        )
    )
    print(
        "Random Forest val accuracy: {}".format(
            np.mean(
                np.where((y_val).astype(np.int) == y_forest_val.astype(np.int), 1, 0)
            )
        )
    )

    acc_svm_train = []
    acc_svm_val = []
    acc_forest_train = []
    acc_forest_val = []

    f1_svm_train = []
    f1_svm_val = []
    f1_forest_train = []
    f1_forest_val = []

    class_freq_svm = []
    class_freq_forest = []
    class_freq_val = []

    for c in range(np.max(y_train) + 1):
        tp = np.sum(np.logical_and(y_train == c, y_svm_train == c))
        fp = np.sum(np.logical_and(y_train != c, y_svm_train == c))
        fn = np.sum(np.logical_and(y_train == c, y_svm_train != c))
        acc_svm_train.append(tp / np.sum(y_train == c))
        f1_svm_train.append(tp / (tp + 0.5 * (fp + fn)))

        tp = np.sum(np.logical_and(y_val == c, y_svm_val == c))
        fp = np.sum(np.logical_and(y_val != c, y_svm_val == c))
        fn = np.sum(np.logical_and(y_val == c, y_svm_val != c))
        acc_svm_val.append(tp / np.sum(y_val == c))
        f1_svm_val.append(tp / (tp + 0.5 * (fp + fn)))
        class_freq_svm.append(np.sum(y_svm_val == c) / len(y_svm_val))

        tp = np.sum(np.logical_and(y_train == c, y_forest_train == c))
        fp = np.sum(np.logical_and(y_train != c, y_forest_train == c))
        fn = np.sum(np.logical_and(y_train == c, y_forest_train != c))
        acc_forest_train.append(tp / np.sum(y_train == c))
        f1_forest_train.append(tp / (tp + 0.5 * (fp + fn)))

        tp = np.sum(np.logical_and(y_val == c, y_forest_val == c))
        fp = np.sum(np.logical_and(y_val != c, y_forest_val == c))
        fn = np.sum(np.logical_and(y_val == c, y_forest_val != c))
        acc_forest_val.append(tp / np.sum(y_val == c))
        f1_forest_val.append(tp / (tp + 0.5 * (fp + fn)))
        class_freq_forest.append(np.sum(y_forest_val == c) / len(y_forest_val))

        class_freq_val.append(np.sum(y_val == c) / len(y_val))

    print("SVM train balanced accuracy: {}".format(np.mean(acc_svm_train)))
    print("SVM val balanced accuracy: {}".format(np.mean(acc_svm_val)))
    print("Random Forest train balanced accuracy: {}".format(np.mean(acc_forest_train)))
    print("Random Forest val balanced accuracy: {}".format(np.mean(acc_forest_val)))

    print("SVM train f1 score: {}".format(np.mean(f1_svm_train)))
    print("SVM val f1 score: {}".format(np.mean(f1_svm_val)))
    print("Random Forest train f1 score: {}".format(np.mean(f1_forest_train)))
    print("Random Forest val f1 score: {}".format(np.mean(f1_forest_val)))

    print(
        "SVM class ratio: {}".format(
            "/".join(["{:.2f}".format(x * 100) + "%" for x in class_freq_svm])
        )
    )
    print(
        "Random Forest class ratio: {}".format(
            "/".join(["{:.2f}".format(x * 100) + "%" for x in class_freq_forest])
        )
    )
    print(
        "True class ratio: {}".format(
            "/".join(["{:.2f}".format(x * 100) + "%" for x in class_freq_val])
        )
    )

    # ----------------------------------------------------------------------------------------
    # Neural Network

    for phase in dataset.keys():
        revtexts, aspects, sentiment = dataset[phase]
        sentiments = SentimentDataset(revtexts, aspects, sentiment)
        dataloaders[phase] = DataLoader(
            sentiments, batch_size=32, shuffle=True, drop_last=False
        )

    length = 0
    for batch in dataloaders["val"]:
        length += torch.sum(batch[-1] == 0)

    device = "cuda" if is_available() else "cpu"
    model = SentimentModel(
        len(dict_for) + 1,
        output_size=3,
        model_name="LSTM_classifier",
        device=device,
        normalize_output=False,
        n_layers=1,
        embedding_dim=200,
        hidden_dim=400,
        dropout=0.1,
        bidirectional=False,
    )
    optimizer = Adam(model.parameters(), lr=0.004, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, cooldown=10, factor=0.4, verbose=True)
    criterion = cross_entropy(dataloaders["train"], device=device)
    train_sentiment_model(model, optimizer, dataloaders, criterion=criterion,
                          scheduler=scheduler, n_epochs=400, eval_every=1)
    model.load_state_dict(
        torch.load(os.path.join("models", model.name + "_best" + ".pth"))[
            "model_state_dict"
        ]
    )

    for metric in [accuracy, class_balanced_accuracy, f1_score]:
        evaluate_sentiment_model(model, dataloaders, criterion=metric)
    class_ratio(model, dataloaders)
