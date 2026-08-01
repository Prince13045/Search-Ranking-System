import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity

from src.components.predection import Prediction


class PredictionPipeline:

    def __init__(self):

        prediction = Prediction()

        self.model = prediction.model
        self.preprocessor = prediction.preprocessor
        self.vectorizer = prediction.vectorizer

    def predict(self, query):

        products = pd.read_csv("artifacts/products.csv")

        query_vector = self.vectorizer.transform([query])

        product_vectors = self.vectorizer.transform(
            products["product_title"].astype(str)
        )

        similarity = cosine_similarity(
            query_vector,
            product_vectors
        ).flatten()

        products["text_similarity"] = similarity

        candidates = products.nlargest(
            100,
            "text_similarity"
        ).copy()

        candidates["query"] = query

        candidates["query_length"] = len(query.split())

        candidates["title_length"] = (
            candidates["product_title"]
            .astype(str)
            .str.split()
            .str.len()
        )

        candidates["membership"] = "Prime"
        candidates["country"] = "India"
        candidates["device_type"] = "Desktop"

        candidates["previous_purchase_count"] = 50
        candidates["average_order_value"] = 25000

        candidates["bm25_score"] = candidates["text_similarity"]

        candidates["query_product_relevance"] = (
            candidates["bm25_score"] +
            candidates["text_similarity"]
        ) / 2

        candidates["rank_position"] = np.arange(
            1,
            len(candidates) + 1
        )

        feature_columns = [
            "previous_purchase_count",
            "average_order_value",
            "price",
            "discount_percent",
            "stock_available",
            "rating",
            "total_reviews",
            "lifetime_sales",
            "bm25_score",
            "query_product_relevance",
            "rank_position",
            "query_length",
            "title_length",
            "text_similarity",
            "membership",
            "country",
            "device_type",
            "brand",
            "category"
        ]

        X = self.preprocessor.transform(
            candidates[feature_columns]
        )

        candidates["ranking_score"] = self.model.predict(X)

        candidates = candidates.sort_values(
            "ranking_score",
            ascending=False
        )

        return candidates[
            [
                "product_title",
                "brand",
                "category",
                "price",
                "rating",
                "ranking_score"
            ]
        ].head(10)


if __name__ == "__main__":

    pipeline = PredictionPipeline()

    result = pipeline.predict("gaming laptop")

    print(result)