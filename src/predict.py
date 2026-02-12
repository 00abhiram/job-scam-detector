import joblib
from preprocess import clean_text

model = joblib.load("models/scam_model.pkl")

def predict_text(text: str):
    cleaned = clean_text(text)
    proba = model.predict_proba([cleaned])[0]
    pred = model.predict([cleaned])[0]

    scam_prob = proba[1]

    if scam_prob < 0.4:
        risk = "LOW"
    elif scam_prob < 0.7:
        risk = "MEDIUM"
    else:
        risk = "HIGH"

    return {
        "prediction": "SCAM" if pred == 1 else "LEGIT",
        "scam_probability": float(scam_prob),
        "risk_level": risk
    }

if __name__ == "__main__":
    text = "Urgent hiring! Pay fee and join"
    print(predict_text(text))