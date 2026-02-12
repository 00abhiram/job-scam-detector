import streamlit as st
import sys
sys.path.append("src")

from predict import predict_text

st.set_page_config(page_title="Job Scam Detector")

st.title("🕵️ Job & Internship Scam Detector")

user_input = st.text_area("Paste the job/internship post here:")

if st.button("Check"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        result = predict_text(user_input)

        st.subheader("Result")
        st.write("Prediction:", result["prediction"])
        st.write("Scam Probability:", round(result["scam_probability"] * 100, 2), "%")
        st.write("Risk Level:", result["risk_level"])