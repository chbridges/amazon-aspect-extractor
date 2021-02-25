import pandas as pd
from utils.aspectextraction import extract_aspects
from utils.reviewextractor import extract_reviews_for_products
from utils.sentiment import SentimentModel
from utils.preprocessing import (
    remove_stopwords_list,
    remove_special_characters_str,
)
import os

from utils.keywordfilter import Filter
import torch

BEST_MODEL_PATH = os.path.join(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "models"),
    "laptops_best.pth",
)


class Pipeline:
    def __init__(
        self,
        algorithm="rake",
        filter_keywords=True,
        aggregate_similar_aspects=True,
        filter_threshold=0.3,
        model_path=BEST_MODEL_PATH,
        **kwargs
    ):
        self.algorithm = algorithm
        self.filter_keywords = filter_keywords
        self.aggregate_similar_aspects = aggregate_similar_aspects
        self.filter_threshold = filter_threshold
        model_dict = torch.load(
            model_path, map_location=torch.device(kwargs.get("device"))
        )
        self.dict_for, self.dict_back = model_dict["dict_for"], model_dict["dict_back"]

        self.sentiment_model = SentimentModel(len(self.dict_for) + 1, **kwargs)
        self.sentiment_model.load_state_dict(model_dict["model_state_dict"])
        self.sentiment_dict = {0: "negative", 1: "neutral", 2: "positive"}
        self.kw_filter = Filter()
        self.kw_filter.load_model()
        self.verbose = kwargs.get("verbose")

    def __call__(self, url: str) -> pd.core.frame.DataFrame:
        # Step 1: Scrape reviews
        if self.verbose:
            print("Fetching reviews from {}".format(url))
        reviews = extract_reviews_for_products([url], 5, 5)[0]

        if self.verbose:
            print("Found {} reviews".format(len(reviews)))

        keyphrases = []
        aspect_masks = []
        aspects = []

        # Step 2: Extract sentences containing aspects
        if self.verbose:
            print("Extracting keywords...")
        for review in reviews:
            kp_temp, am_temp = extract_aspects(review, algorithm=self.algorithm)
            keyphrases.extend(kp_temp)
            aspect_masks.extend(am_temp)

        # Step 3: Recreate aspects from keyphrases and apply filter
        for i in range(len(keyphrases)):
            # Recreate aspect from keyphrase and aspect mask
            aspect = [
                keyphrases[i][j]
                for j in range(len(keyphrases[i]))
                if aspect_masks[i][j] == 1
            ]
            aspect = " ".join(aspect)

            aspects.append(aspect)
        if self.filter_keywords:
            if self.verbose:
                print("Filtering keywords...")

            filter_mask = list(
                map(
                    lambda x: x > self.filter_threshold, self.kw_filter.predict(aspects)
                )
            )

            aspects = [aspects[i] for i in range(len(filter_mask)) if filter_mask[i]]
            keyphrases = [
                keyphrases[i] for i in range(len(filter_mask)) if filter_mask[i]
            ]
            aspect_masks = [
                aspect_masks[i] for i in range(len(filter_mask)) if filter_mask[i]
            ]
        aspect_texts = aspects.copy()

        seq_lens = [len(rev) for rev in keyphrases]

        # Step 3.5: Preprocessing for the neural network
        if self.verbose:
            print("Preprocessing keywords...")
        for i in range(len(keyphrases)):
            keyphrases[i], aspect_masks[i] = remove_stopwords_list(
                keyphrases[i], aspect_masks[i]
            )
            keyphrases[i] = [token.lower() for token in keyphrases[i]]
            keyphrases[i] = list(map(remove_special_characters_str, keyphrases[i]))
            keyphrases[i] = [self.dict_for.get(token) for token in keyphrases[i]]
            keyphrases[i] = [
                token if not token is None else 0 for token in keyphrases[i]
            ]

        max_len = max(seq_lens)

        [rev.extend([0] * (max_len - len(rev))) for rev in keyphrases]
        [asp.extend([0] * (max_len - len(asp))) for asp in aspect_masks]
        seq_lens = [max_len] * len(keyphrases)
        # Step 4: Predict sentiments
        # Predict sentiment
        if self.verbose:
            print("Predicting sentiment...")
        self.sentiment_model.init_hidden(len(keyphrases))
        keyphrases = (
            torch.Tensor(keyphrases).to(self.sentiment_model.device).type(torch.long)
        )
        aspect_masks = (
            torch.Tensor(aspect_masks).to(self.sentiment_model.device).type(torch.long)
        )
        x = torch.stack((keyphrases, aspect_masks), dim=-1)
        pred_scores = self.sentiment_model(
            x, torch.Tensor(seq_lens).to(self.sentiment_model.device)
        )
        pred_classes = torch.argmax(pred_scores, dim=-1)
        sentiments = [self.sentiment_dict[i.item()] for i in pred_classes]

        if self.verbose:
            print("Creating Dataframe...")
        data = []
        for i, aspect in enumerate(aspect_texts):
            data.append([aspect, sentiments[i], pred_classes[i].item()])

        # Create dataframe of (aspect, sentiment) tuples
        df = pd.DataFrame(data, columns=["aspect", "sentiment_text", "sentiment_value"])

        # Step 5: Aggregate similar aspects
        if self.aggregate_similar_aspects:
            pass

        # Step 6: Group df.aspect by count and average sentiment, keep top k aspects
        df = (
            df.groupby(["aspect", "sentiment_text"])
            .size()
            .reset_index(name="counts")
            .sort_values("counts", ascending=False)
        )

        # Step 7: Split df into 3 dataframes? (positive, neutral, negative)
        df_pos = df.loc[df["sentiment_text"] == "positive"]
        df_neu = df.loc[df["sentiment_text"] == "neutral"]
        df_neg = df.loc[df["sentiment_text"] == "negative"]

        data = {"all": df, "pos": df_pos, "neu": df_neu, "neg": df_neg}

        return data
