import os
import sys
import yaml
import joblib

from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBRanker

from src.logger import logging
from src.exception import custom_exception
from src.components.feature_engennering import FeatureEngineering 


class ModelTrainer:

    def __init__(self):

        with open("config/config.yaml", "r") as file:
            config = yaml.safe_load(file)

        self.random_state = config["model"]["random_state"]
        self.test_size = config["model"]["test_size"]

    def create_relevance(self, df):

        logging.info("Creating relevance labels")

        df["relevance"] = 0

        df.loc[df["clicked"] == 1, "relevance"] = 1
        df.loc[df["added_to_cart"] == 1, "relevance"] = 2
        df.loc[df["purchased"] == 1, "relevance"] = 3

        return df

    def initiate_model_training(self):

        try:

            logging.info("Starting Feature Engineering")

            fe = FeatureEngineering()

            df = fe.initiate_feature_engineering()

            df = self.create_relevance(df)

            logging.info("Splitting Dataset")

            splitter = GroupShuffleSplit(
                test_size=self.test_size,
                random_state=self.random_state
            )

            train_idx, test_idx = next(
                splitter.split(
                    df,
                    groups=df["session_id"]
                )
            )

            train_df = df.iloc[train_idx].copy()
            test_df = df.iloc[test_idx].copy()

            logging.info(f"Train Shape : {train_df.shape}")
            logging.info(f"Test Shape : {test_df.shape}")

            feature_columns = (
                fe.numerical_columns +
                fe.categorical_columns
            )

            preprocessor = fe.get_preprocessor()

            logging.info("Transforming Training Data")

            X_train = preprocessor.fit_transform(
                train_df[feature_columns]
            )

            logging.info("Transforming Testing Data")

            X_test = preprocessor.transform(
                test_df[feature_columns]
            )

            y_train = train_df["relevance"]
            y_test = test_df["relevance"]

            train_group = (
                train_df
                .groupby("session_id")
                .size()
                .to_numpy()
            )

            test_group = (
                test_df
                .groupby("session_id")
                .size()
                .to_numpy()
            )

            logging.info("Training XGBRanker")

            model = XGBRanker(

                objective="rank:ndcg",

                learning_rate=0.1,

                n_estimators=300,

                max_depth=6,

                subsample=0.8,

                colsample_bytree=0.8,

                random_state=self.random_state

            )

            model.fit(

                X_train,

                y_train,

                group=train_group,

                eval_set=[(X_test, y_test)],

                eval_group=[test_group],

                verbose=True

            )

            logging.info("Predicting Ranking Scores")

            scores = model.predict(X_test)

            test_df["ranking_score"] = scores

            ranked_results = test_df.sort_values(

                ["session_id", "ranking_score"],

                ascending=[True, False]

            )

            os.makedirs("artifacts", exist_ok=True)

            joblib.dump(

                model,

                "artifacts/xgb_ranker.pkl"

            )

            joblib.dump(

                preprocessor,

                "artifacts/preprocessor.pkl"

            )

            ranked_results.to_csv(

                "artifacts/ranked_results.csv",

                index=False

            )

            logging.info("Model Saved Successfully")

            logging.info("Preprocessor Saved Successfully")

            logging.info("Ranking Results Saved Successfully")

            print("\nTraining Completed Successfully")

            print(ranked_results.head())

            return model

        except Exception as e:

            raise custom_exception(e, sys)


if __name__ == "__main__":

    trainer = ModelTrainer()

    trainer.initiate_model_training()