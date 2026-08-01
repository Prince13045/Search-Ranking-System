import joblib

class Prediction:

    def __init__(self):

        self.model = joblib.load("artifacts/xgb_ranker.pkl")

        self.preprocessor = joblib.load("artifacts/preprocessor.pkl")

        self.vectorizer = joblib.load("artifacts/tfidf_vectorizer.pkl")