import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Add this block right after imports
try:
    # Try to download NLTK data
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception as e:
    # If download fails, try setting a custom download path
    import os
    nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
    os.makedirs(nltk_data_dir, exist_ok=True)
    nltk.data.path.append(nltk_data_dir)
    nltk.download('punkt', download_dir=nltk_data_dir, quiet=True)
    nltk.download('stopwords', download_dir=nltk_data_dir, quiet=True)

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)

# Load the vectorizer and model
with open('vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)
with open('model.pkl', 'rb') as f:
    mnb = pickle.load(f)

# Streamlit UI
st.title("Spam Classifier Web App")
input_sms = st.text_area("Enter the message:")

if st.button('Predict'):
    # 1. Preprocess
    transformed_sms = transform_text(input_sms)
    # 2. Vectorize
    vector_input = tfidf.transform([transformed_sms]).toarray()
    # 3. Predict
    result = mnb.predict(vector_input)[0]
    # 4. Output
    if result == 1:
        st.error("Spam")
    else:
        st.success("Not Spam")
