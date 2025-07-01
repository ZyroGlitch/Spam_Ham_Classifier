import streamlit as st
import joblib

model = joblib.load('models/spam_classifier_model.pkl')
vectorizer = joblib.load('models/spam_vectorizer_model.pkl')

st.title("SMS Spam Classifier")
st.write("Enter your SMS message below to check if it is spam or ham.")

user_input = st.text_area("Type your message here: ", "")

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter a message to classify.")
    else:
        vectorized_input = vectorizer.transform([user_input])
        prediction = model.predict(vectorized_input)[0]
        prediction_prob = model.predict_proba(vectorized_input).max()

        if prediction == "spam":
            st.error(f"Prediction: **SPAM** ({prediction_prob:.2f} confidence)")
        else:
            st.success(f"Prediction: **HAM** ({prediction_prob:.2f} confidence)")

