import re
from sklearn.feature_extraction.text import TfidfVectorizer

money_words = ["pay", "fee", "salary", "rs", "inr", "dollar", "money"]
urgency_words = ["urgent", "immediate", "limited", "hurry", "fast"]

def extra_features(texts):
    features = []
    for text in texts:
        text_lower = text.lower()
        money_count = sum(word in text_lower for word in money_words)
        urgency_count = sum(word in text_lower for word in urgency_words)
        url_count = len(re.findall(r"http\S+|www\S+", text_lower))
        phone_present = 1 if re.search(r"\d{10}", text_lower) else 0

        features.append([money_count, urgency_count, url_count, phone_present])
    return features

def build_vectorizer():
    return TfidfVectorizer(ngram_range=(1,2), max_features=5000)

def extra_features_wrapper(x):
    return extra_features(x)