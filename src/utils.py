import joblib

def save_object(path, obj):
    joblib.dump(obj, path)

def load_object(path):
    return joblib.load(path)