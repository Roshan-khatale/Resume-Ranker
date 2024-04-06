import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

resumes = pd.read_csv("data/resumes.csv")
with open("data/job_description.txt") as f:
    jd = f.read()

docs = resumes["resume"].tolist()
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(docs + [jd])
scores = cosine_similarity(X[-1], X[:-1])[0]

resumes["score"] = scores
resumes = resumes.sort_values("score", ascending=False)
print(resumes[["name", "score"]])