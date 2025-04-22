def save_model(model, path):
    import joblib
    joblib.dump(model, path)

def load_model(path):
    import joblib
    return joblib.load(path)
