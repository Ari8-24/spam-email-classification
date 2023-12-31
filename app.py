import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer

porter_stemmer = PorterStemmer()

def modify_text(text):
    # lower casing
    text = text.lower()

    # tokenization
    text = nltk.word_tokenize(text)

    # special charecter removal
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    # stop words/punctuations removal
    text = y[:]  # cloning to reassign text
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    # stemming
    text = y[:]  # cloning to reassign text
    y.clear()

    for i in text:
        y.append(porter_stemmer.stem(i))

    return " ".join(y)


tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Spam Email Classifier")

#accept user input
input = st.text_area("Enter the email you want to test")

if st.button('Predict'):

    #Preprocess
    modified_email = modify_text(input)

    #Vectorize
    vectorized_input = tfidf.transform([modified_email])

    #predict
    result = model.predict(vectorized_input)[0]

    #Output
    if result == 1:
        st.header("Spam :(")
    else:
        st.header("Ham :)")


