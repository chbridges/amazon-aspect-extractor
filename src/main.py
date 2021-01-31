import os

import numpy as np
import torch
from torch.cuda import is_available
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from utils.baselines import SentimentForest, SentimentSVM, top_ngrams
from utils.dataloading import load_semeval2015
from utils.metrics import accuracy, class_balanced_accuracy, cross_entropy, f1_score
from utils.preprocessing import PreprocessingPipeline
from utils.sentiment import SentimentDataset, SentimentModel, evaluate_sentiment_model
from utils.reviewextractor import extract_reviews_for_products

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
    
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/ABSA15")
    dataset = load_semeval2015(path, categories=["restaurants"])
    pipeline = PreprocessingPipeline()
    dataloaders = {}

    dataset, dict_for, dict_back = pipeline(dataset)
    revtexts, aspects, sentiment = dataset["train"]
    train_set = SentimentDataset(revtexts, aspects, sentiment)

    revtexts, aspects, sentiment = dataset["val"]
    val_set = SentimentDataset(revtexts, aspects, sentiment)
    # ----------------------------------------------------------------------------------------
    # Baseline Models
    top_x = top_ngrams(train_set)
    forest = SentimentForest(top_x)
    svm = SentimentSVM(top_x)

    x_train = torch.stack([event for event, _, _ in train_set], axis=0).numpy()
    y_train = torch.Tensor([sentiment for _, _, sentiment in train_set]).numpy()
    y_train = (2 * y_train).astype(np.int)

    x_val = torch.stack([event for event, _, _ in val_set], axis=0).numpy()
    y_val = torch.Tensor([sentiment for _, _, sentiment in val_set]).numpy()
    y_val = (2 * y_val).astype(np.int)

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

    # ----------------------------------------------------------------------------------------
    # Neural Network

    # for phase in dataset.keys():
    #     revtexts, aspects, sentiment = dataset[phase]
    #     sentiments = SentimentDataset(revtexts, aspects, sentiment)
    #     dataloaders[phase] = DataLoader(sentiments, batch_size=32, shuffle=True)
    #
    # device = "cuda" if is_available() else "cpu"
    # model = SentimentModel(
    #     len(dict_for) + 1,
    #     output_size=3,
    #     model_name="LSTM_classifier",
    #     device=device,
    #     normalize_output=False,
    #     n_layers=1,
    #     embedding_dim=128,
    #     hidden_dim=768,
    #     dropout=0.4,
    #     bidirectional=False,
    # )
    # optimizer = Adam(model.parameters(), lr=0.004, weight_decay=1e-5)
    # scheduler = ReduceLROnPlateau(optimizer, cooldown=10, factor=0.2, verbose=True)
    # criterion = cross_entropy(dataloaders["train"], device=device)
    # # train_sentiment_model(model, optimizer, dataloaders, criterion=criterion,
    # #                       scheduler=scheduler, n_epochs=150, eval_every=1)
    # model.load_state_dict(
    #     torch.load(os.path.join("models", model.name + "_best" + ".pth"))[
    #         "model_state_dict"
    #     ]
    # )
    # for metric in [accuracy, class_balanced_accuracy, f1_score]:
    #     evaluate_sentiment_model(model, dataloaders, criterion=metric)
