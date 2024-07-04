# preprocess.py

import re
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Load Spacy's English language model
nlp = spacy.load('en_core_web_sm')

# Define stop words
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """
    Function to clean text by removing URLs, special characters, and extra spaces.
    """
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_text(text):
    """
    Function to tokenize text into words.
    """
    tokens = word_tokenize(text)
    return tokens

def remove_stopwords(tokens):
    """
    Function to remove stop words from a list of tokens.
    """
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    return filtered_tokens

def lemmatize_tokens(tokens):
    """
    Function to lemmatize a list of tokens.
    """
    lemmatized_tokens = [nlp(word)[0].lemma_ for word in tokens]
    return lemmatized_tokens

def preprocess_text(text):
    """
    Complete preprocessing pipeline: clean, tokenize, remove stopwords, and lemmatize.
    """
    text = clean_text(text)
    tokens = tokenize_text(text)
    tokens = remove_stopwords(tokens)
    lemmatized_text = ' '.join(lemmatize_tokens(tokens))
    return lemmatized_text

# Example usage
if __name__ == "__main__":
    sample_text = "This is a sample text for preprocessing! Visit https://example.com for more details. #NLP @spacy"
    print(preprocess_text(sample_text))
