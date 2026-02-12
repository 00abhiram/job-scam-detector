import re
import spacy
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")
nlp = spacy.load("en_core_web_sm")

stop_words = set(stopwords.words("english"))

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    doc = nlp(text)
    tokens = []
    for token in doc:
        if token.text not in stop_words:
            tokens.append(token.lemma_)

    return " ".join(tokens)

if __name__ == "__main__":
    sample = "Urgent!!! Pay ₹500 now to join http://fake.com"
    print(clean_text(sample))