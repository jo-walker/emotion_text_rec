import re # regular expressions for text cleaning
import nltk # Natural Language ToolKit for tokenizing and stop words
import spacy # For advanced natural language processing
from nltk.corpus import stopwords # NLTK module for removing common words
from nltk.tokenize import word_tokenize # and tokenizing text

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
    sample_text = """
    Dear [Company Name] Support Team I hope this message finds you well. I am writing to bring to your attention a significant issue I encountered with the latest update to your software. After installing the most recent update (version 2.3.1 I noticed that a critical feature of the software which I rely on daily is no longer functioning correctly. Specifically the data export function now generates files with incorrect formatting making the data unusable for my reports. This problem started immediately after the update and I verified that it was not present in the previous version (2.3.0). To make matters worse I have a presentation scheduled with my team tomorrow morning and this error has put me in a very difficult position. I spent several hours troubleshooting and attempting various workarounds but to no avail. It is quite embarrassing considering I have always advocated for your software within my organization. I would appreciate it if your development team could look into this matter urgently and provide a fix or a rollback option to the previous stable version. Additionally any immediate advice on how to resolve the issue temporarily would be greatly appreciated. I trust that your team will handle this matter with the urgency it requires. Thank you for your prompt attention to this issue. Best regards
    """
    print(preprocess_text(sample_text))
