import os 
import sys
import yaml
import pandas as pd
from src.logger import logging
from src.exception import custom_exception

class Datavalidation:
    def __init__(self):
        config_path = os.path.join("config", "config.yaml")
        with open(config_path, "r") as file:
            config=yaml.safe_load(file)
        self.raw_data_path=config["data"]["raw_data_path"]
        self.processed_data_path=config["data"]["processed_data_path"]

    def validate_data(self):
        try:
            logging.info("loading_raw_data")
            df=pd.read_csv(self.raw_data_path)
            logging.info(f"initial_shape:{df.shape}")
            #missing_values
            missing=df.isnull().sum()
            logging.info(f"missing_values:{missing}")
            #Remove duplicate
            duplicate = df.duplicated().sum()
            logging.info(f"Duplicate rows: {duplicate}")
            df = df.drop_duplicates()
            #price validation
            df=df[df["price"]>0]
            #rating_validation
            df=df[(df['rating']>=0)&(df['rating']<=5)]
            #Reviews validation
            df=df[df['total_reviews']>0]
            #sales validation
            df=df[df["lifetime_sales"]>=0]
            for col in [
                "clicked",
                "added_to_cart",
                "purchased"
            ]:
                df[col]=df[col].astype(int)
            logging.info(f"data validate:{df.shape}")
            os.makedirs(
                os.path.dirname(self.processed_data_path),
                exist_ok=True
            )
            df.to_csv(
                self.processed_data_path,
                index=False
            )
            logging.info("processed data path saved sucessfully")
            return df
        except Exception as e:
            raise custom_exception(e,sys)

if __name__ == "__main__":

    validator = Datavalidation()

    df = validator.validate_data()

    print(df.head())

    print(df.shape)