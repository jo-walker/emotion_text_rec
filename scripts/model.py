# model.py

from load_data import load_datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib

# Load and preprocess datasets
train_df, test_df, val_df = load_datasets('emotion_dataset/training.csv', 'emotion_dataset/test.csv', 'emotion_dataset/validation.csv')
X_train = train_df['cleaned_text']
y_train = train_df['emotion']
X_test = test_df['cleaned_text']
y_test = test_df['emotion']
X_val = val_df['cleaned_text']
y_val = val_df['emotion']

# Pipeline
model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('classifier', MultinomialNB())
])

# Train model
model.fit(X_train, y_train)

# Predict on test data
y_pred_test = model.predict(X_test)

# Evaluate on test data
print("Test Data Evaluation:")
print(classification_report(y_test, y_pred_test))

# Predict on validation data
y_pred_val = model.predict(X_val)

# Evaluate on validation data
print("Validation Data Evaluation:")
print(classification_report(y_val, y_pred_val))

# Save model
joblib.dump(model, 'models/emotion_model.pkl')
