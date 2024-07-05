# model.py: This script trains and evaluates a machine learning model for emotion detection in text.
# Loads and preprocesses data, constructs a pipeline, trains a Multinomial Naive Bayes model, evaluates the model, and saves the trained model.

from load_data import load_datasets # Function from load_data.py to load and preprocess datasets.
from sklearn.feature_extraction.text import TfidfVectorizer # Converts text data into numerical features using TF-IDF.
from sklearn.naive_bayes import MultinomialNB # Multinomial Naive Bayes classifier.
from sklearn.pipeline import Pipeline # Creates a pipeline to streamline the process of transforming data and applying a model.
from sklearn.metrics import classification_report # Creates a pipeline to streamline the process of transforming data and applying a model.
import joblib # For saving and loading the trained model.

# Load and preprocess datasets
train_df, test_df, val_df = load_datasets('emotion_dataset/training.csv', 'emotion_dataset/test.csv', 'emotion_dataset/validation.csv') #Loads and preprocesses the datasets.
X_train = train_df['cleaned_text'] #Splits data into features (X) and labels (y).
y_train = train_df['emotion']
X_test = test_df['cleaned_text']
y_test = test_df['emotion']
X_val = val_df['cleaned_text']
y_val = val_df['emotion']

# Pipeline: Creates a pipeline with TF-IDF vectorization and the Multinomial Naive Bayes classifier.
model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('classifier', MultinomialNB())
])

# Train model: Trains the model on the training data.
model.fit(X_train, y_train)

# Predict on test data using classification reports.
y_pred_test = model.predict(X_test)

# Evaluate on test data using classification reports.
print("Test Data Evaluation:")
print(classification_report(y_test, y_pred_test))

# Predict on validation data using classification reports.
y_pred_val = model.predict(X_val)

# Evaluate on validation datausing classification reports.
print("Validation Data Evaluation:")
print(classification_report(y_val, y_pred_val))

# Save model: Saves the trained model to a file for future use.
joblib.dump(model, 'models/emotion_model.pkl')
