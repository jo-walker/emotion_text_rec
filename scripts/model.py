from load_data import load_datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib

# Load and preprocess datasets
train_df, test_df, val_df, single_text_df = load_datasets(
    'emotion_dataset/training.csv', 
    'emotion_dataset/test.csv', 
    'emotion_dataset/validation.csv',
    'emotion_dataset/unlabeled_text.csv'
)

X_train = train_df['cleaned_text']
y_train = train_df['label']
X_test = test_df['cleaned_text']
y_test = test_df['label']
X_val = val_df['cleaned_text']
y_val = val_df['label']
X_unlabeled = single_text_df['cleaned_text']

def train_and_evaluate(model_name, model):
    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)
    print("Test Data Evaluation:")
    print(classification_report(y_test, y_pred_test))

    y_pred_val = model.predict(X_val)
    print("Validation Data Evaluation:")
    print(classification_report(y_val, y_pred_val))

    joblib.dump(model, f'models/{model_name}.pkl')

    # Prediction for the unlabeled text
    y_pred_unlabeled = model.predict(X_unlabeled)
    print(f"Predictions for unlabeled text: {y_pred_unlabeled}")

models = {
    'Naive Bayes': Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('classifier', MultinomialNB())
    ]),
    'SVM': Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('classifier', LinearSVC())
    ]),
    'Logistic Regression': Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('classifier', LogisticRegression())
    ])
}

for model_name, model in models.items():
    print(f"\nTraining and evaluating model: {model_name}")
    train_and_evaluate(model_name, model)
