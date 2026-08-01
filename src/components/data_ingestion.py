import os
import sys
import yaml
import pandas as pd
from urllib.parse import quote_plus
from sqlalchemy import create_engine

from src.exception import custom_exception
from  src.logger import logging


class DataIngestion:

    def __init__(self):
        print("Current Working Directory:", os.getcwd())
        print("Config exists:", os.path.exists("config/config.yaml"))
        with open("config/config.yaml", "r") as file:
            config = yaml.safe_load(file)

        self.host = config["database"]["host"]
        self.port = config["database"]["port"]
        self.username = config["database"]["username"]
        self.password = config["database"]["password"]
        self.database = config["database"]["database"]

        self.raw_data_path = config["data"]["raw_data_path"]

    def initiate_data_ingestion(self):

        try:

            logging.info("Connecting to MySQL database")
            encoded_password = quote_plus(self.password)
            engine = create_engine(
                f"mysql+pymysql://{self.username}:{encoded_password}@{self.host}:{self.port}/{self.database}"
            )

            query = """
            SELECT

            u.user_id,
            u.membership,
            u.country,
            u.previous_purchase_count,
            u.average_order_value,

            ss.session_id,
            ss.query,
            ss.device_type,
            ss.search_time,

            p.product_id,
            p.product_title,
            p.brand,
            p.category,
            p.price,
            p.discount_percent,
            p.stock_available,
            p.rating,
            p.total_reviews,
            p.lifetime_sales,

            sr.rank_position,
            sr.bm25_score,
            sr.tfidf_similarity,
            sr.query_product_relevance,

            ua.clicked,
            ua.added_to_cart,
            ua.purchased

            FROM search_sessions ss

            JOIN users u
            ON ss.user_id=u.user_id

            JOIN search_results sr
            ON ss.session_id=sr.session_id

            JOIN products p
            ON sr.product_id=p.product_id

            JOIN user_actions ua
            ON sr.session_id=ua.session_id
            AND sr.product_id=ua.product_id;
            """

            logging.info("Reading data from MySQL")

            df = pd.read_sql(query, engine)

            os.makedirs(os.path.dirname(self.raw_data_path), exist_ok=True)

            logging.info("Saving raw dataset")

            df.to_csv(self.raw_data_path, index=False)

            logging.info(f"Dataset Shape : {df.shape}")

            return df

        except Exception as e:
            raise custom_exception(e, sys)


if __name__ == "__main__":

    ingestion = DataIngestion()

    df = ingestion.initiate_data_ingestion()

    print(df.head())

    print(df.shape)