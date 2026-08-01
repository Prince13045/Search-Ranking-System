import os
import sys
import yaml
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

from src.logger import logging
from src.exception import custom_exception


class FeatureEngineering:

    def __init__(self):

        with open("config/config.yaml", "r") as file:
            config = yaml.safe_load(file)

        self.processed_data_path = config["data"]["processed_data_path"]
        self.numerical_columns = config["feature_engineering"]["numerical_columns"]
        self.categorical_columns = config["feature_engineering"]["categorical_columns"]

    def create_text_similarity(self, df):

        corpus = pd.concat(
            [
                df["query"].astype(str),
                df["product_title"].astype(str)
            ],
            ignore_index=True
        )

        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=5000
        )

        vectorizer.fit(corpus)

        query_vectors = vectorizer.transform(df["query"].astype(str))
        product_vectors = vectorizer.transform(df["product_title"].astype(str))

        similarity = (
            query_vectors.multiply(product_vectors)
            .sum(axis=1)
            .A1
        )

        df["text_similarity"] = similarity

        os.makedirs("artifacts", exist_ok=True)

        joblib.dump(
            vectorizer,
            "artifacts/tfidf_vectorizer.pkl"
        )

        return df

    def get_preprocessor(self):

        numeric_pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler())
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False
                    )
                )
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    numeric_pipeline,
                    self.numerical_columns
                ),
                (
                    "cat",
                    categorical_pipeline,
                    self.categorical_columns
                )
            ],
            remainder="drop"
        )

        return preprocessor

    def initiate_feature_engineering(self):

        try:

            df = pd.read_csv(self.processed_data_path)

            df["query_length"] = (
                df["query"]
                .astype(str)
                .str.split()
                .str.len()
            )

            df["title_length"] = (
                df["product_title"]
                .astype(str)
                .str.split()
                .str.len()
            )

            df = self.create_text_similarity(df)

            preprocessor = self.get_preprocessor()

            feature_columns = (
                self.numerical_columns +
                self.categorical_columns
            )

            preprocessor.fit(df[feature_columns])

            os.makedirs("artifacts", exist_ok=True)

            joblib.dump(
                preprocessor,
                "artifacts/preprocessor.pkl"
            )

            products = df[
                [
                    "product_id",
                    "product_title",
                    "brand",
                    "category",
                    "price",
                    "discount_percent",
                    "stock_available",
                    "rating",
                    "total_reviews",
                    "lifetime_sales"
                ]
            ].drop_duplicates()

            products.to_csv(
                "artifacts/products.csv",
                index=False
            )

            return df

        except Exception as e:
            raise custom_exception(e, sys)


if __name__ == "__main__":

    fe = FeatureEngineering()

    df = fe.initiate_feature_engineering()

    print(df.head())

    print(df.shape)