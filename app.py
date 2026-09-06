import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download NLTK stopwords
nltk.download("stopwords", quiet=True)

# Load model and vectorizer
model = joblib.load("model.joblib")
tfidf = joblib.load("vectorizer.pkl")

# Text preprocessing
ps = PorterStemmer()
stop_words = set(stopwords.words("english"))

def transform_text(text):
    text = text.lower()
    text = re.sub("[^a-zA-Z]", " ", text)
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

# Streamlit app
st.title("📱 SMS Spam Detection")

st.write("Enter an SMS message to check whether it is Spam or Ham.")

message = st.text_area("Enter your SMS:")

if st.button("Check SMS"):

    if message.strip() == "":
        st.warning("Please enter an SMS message.")

    else:
        processed_message = transform_text(message)

        message_vector = tfidf.transform([processed_message]).toarray()

        prediction = model.predict(message_vector)[0]

        if prediction == 1:
            st.error("🚨 Spam Message")
        else:
            st.success("✅ Ham (Not Spam) Message")
