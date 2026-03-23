import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv("spam_ham_dataset.csv")

# Clean column names
df.columns = df.columns.str.strip()

st.write("Columns:", df.columns)  # Debug

# Change based on your dataset
X = df[df.columns[1]]
y = df[df.columns[0]]

vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(X)

model = MultinomialNB()
model.fit(X_vec, y)

st.title("Spam Classifier")

msg = st.text_input("Enter a message")

if msg:
    msg_vec = vectorizer.transform([msg])
    prediction = model.predict(msg_vec)
    st.write("Prediction:", prediction[0])
