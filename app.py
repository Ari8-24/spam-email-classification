import streamlit as st
import pickle
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer

# Download NLTK resource
nltk.download('punkt')
nltk.download('stopwords')

porter_stemmer = PorterStemmer()

# Define text preprocessing function
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize text bhjnu
    tokens = word_tokenize(text)
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    # Join tokens back into text
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
model = pickle.load(open('naive_bayes_model.pkl', 'rb'))

st.title("Spam Email Classifier")

# Accept user input
input = st.text_area("Enter the text you want to test")

if st.button('Predict'):
    # Preprocess user input
    preprocessed_text = preprocess_text(input)

    # Vectorize the preprocessed text
    vectorized_input = vectorizer.transform([preprocessed_text])

    # Predict the result
    result = model.predict(vectorized_input)[0]

    # Display the result
    if result == 1:
        st.header("Spam :(")
    else:
        st.header("Ham :)")