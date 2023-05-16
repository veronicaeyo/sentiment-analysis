
import pickle
import string
import numpy as np 
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download('punkt')
nltk.download('stopwords') 

nltk.download('wordnet') 
nltk.download('omw-1.4')
lemmatizer = WordNetLemmatizer()


with open("model/tf_idf.pkl", "rb") as f:
    lsa_idf = pickle.load(f)

stop_words = stopwords.words('english') 
punctuation_list = list(string.punctuation)
useless_words = stop_words + punctuation_list

def preprocess(text: str) ->np.array:
    if text:
        test_string = " ".join(word for word in word_tokenize(text.lower()) if word not in useless_words)
        test_string = " ".join(lemmatizer.lemmatize(word) for word in word_tokenize(test_string))
        test_string_idf = lsa_idf.transform([test_string])

        return test_string_idf
    else:
        raise ValueError("Please enter a text")

if __name__ == '__main__':
    preprocess()
