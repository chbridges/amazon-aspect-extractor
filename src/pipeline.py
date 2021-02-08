# from utils.keywordfilter import Filter
import pandas as pd
from utils.aspectextraction import extract_aspects
from utils.keywords import aggregate_similar_keywords
from utils.reviewextractor import extract_reviews_for_products
from utils.sentiment import SentimentModel

from keywordfilter import Filter


class Pipeline:
    def __init__(
        self,
        algorithm="rake",
        filter_keywords=False,
        aggregate_similar_aspects=True,
        filter_threshold=0.3,
        **kwargs
    ):
        self.algorithm = algorithm
        self.filter_keywords = filter_keywords
        self.aggregate_similar_aspects = aggregate_similar_aspects
        self.filter_threshold = filter_threshold
        self.sentiment_model = SentimentModel(**kwargs)

    def __call__(self, url: str) -> pd.core.frame.DataFrame:
        # Step 1: Scrape reviews
        reviews = extract_reviews_for_products(url)

        keyphrases = []
        aspect_masks = []
        aspects = []
        sentiments = []

        # Step 2: Extract sentences containing aspects
        for review in reviews:
            kp_temp, am_temp = extract_aspects(review, algorithm=self.algorithm)
            keyphrases.extend(kp_temp)
            aspect_masks.extract_aspects(am_temp)

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
            kw_filter = Filter()

            filter_mask = list(
                map(lambda x: x > self.filter_threshold, kw_filter.predict(aspects))
            )

            aspects = [aspects[i] for i in range(len(filter_mask)) if filter_mask[i]]
            keyphrases = [
                keyphrases[i] for i in range(len(filter_mask)) if filter_mask[i]
            ]
            aspect_masks = [
                aspect_masks[i] for i in range(len(filter_mask)) if filter_mask[i]
            ]

        # Step 4: Predict sentiments
        for i in range(len(aspects)):
            # Recreate aspect from keyphrase and aspect mask
            aspect = [
                keyphrases[i][j]
                for j in range(len(keyphrases[i]))
                if aspect_masks[i][j] == 1
            ]
            aspect = " ".join(aspect)
            # Predict sentiment
            sentiment = 0  # Armand pls help
            # Store sentiment as (aspect, sentiment) tuple
            sentiments.append(aspect, sentiment)

        # Create dataframe of (aspect, sentiment) tuples
        df = pd.DataFrame(sentiments, columns=["aspect", "sentiment"])

        # Step 5: Aggregate similar aspects
        if self.aggregate_similar_aspects:
            pass

        # Step 6: Group df.aspect by count and average sentiment, keep top k aspects

        # Step 7: Split df into 3 dataframes? (positive, neutral, negative)

        return df
