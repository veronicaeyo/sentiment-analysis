import streamlit as st
import pickle
from scripts.preprocess import preprocess

with open("model/mnb_model.pkl", "rb") as f:
    model = pickle.load(f)

sentiment_dict = {1: "Depressed", 0: "Not Depressed"}

st.title("Sentiment Analysis")


try:
    text = st.text_input("Please enter your text", "")
    if st.button("Predict"):
        test_string = preprocess(text)
        prediction = model.predict(test_string)

        prob = model.predict_proba(test_string).max()
        st.write(
            f"The tone of this sentence is: **{sentiment_dict.get(prediction[0])}**, with probability: {prob:.2f}"
        )
except ValueError as e:
    st.write(str(e))
