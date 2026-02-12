import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer
import joblib

from preprocess import clean_text
from features import extra_features, extra_features_wrapper, build_vectorizer

df = pd.read_csv("data/dataset.csv")
df["clean_text"] = df["text"].astype(str).apply(clean_text)

X = df["clean_text"]
y = df["label"]

tfidf = build_vectorizer()
extra_transformer = FunctionTransformer(extra_features_wrapper, validate=False)

features = FeatureUnion([
    ("tfidf", tfidf),
    ("extra", extra_transformer)
])

model = Pipeline([
    ("features", features),
    ("clf", LogisticRegression(max_iter=1000))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))

joblib.dump(model, "models/scam_model.pkl")
print("Model saved!")