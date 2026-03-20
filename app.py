import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load dataset
df = pd.read_csv("spam.csv")

# Prepare data
X = df['v2']
y = df['v1']

# Train model
vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(X)

model = MultinomialNB()
model.fit(X_vec, y)

# UI
st.title("Spam Classifier")

msg = st.text_input("Enter a message")

if msg:
    msg_vec = vectorizer.transform([msg])
    prediction = model.predict(msg_vec)
    st.write("Prediction:", prediction[0])
