# load_data.py: This script loads datasets from CSV files, preprocesses the text, and returns the cleaned datasets.

import pandas as pd # data manipulation and analysis
from preprocess import preprocess_text # fcn from preprocess.py

def load_datasets(train_path, test_path, val_path):
    """
    Loads the datasets from CSV files, preprocesses the text data, and returns the cleaned datasets.
    """
    # Load datasets
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    val_df = pd.read_csv(val_path)
    
    # Preprocess text
    train_df['cleaned_text'] = train_df['text'].apply(preprocess_text)
    test_df['cleaned_text'] = test_df['text'].apply(preprocess_text)
    val_df['cleaned_text'] = val_df['text'].apply(preprocess_text)
    
    return train_df, test_df, val_df

# main execution
if __name__ == "__main__":
    # loads and preprocesses the training, test, validation datasets.
    train_df, test_df, val_df = load_datasets('emotion_dataset/training.csv', 'emotion_dataset/test.csv', 'emotion_dataset/validation.csv')
    print("Training Data:")
    print(train_df.head())
    print("\nTest Data:")
    print(test_df.head())
    print("\nValidation Data:")
    # prints the first few rows of each dataset to verify the loading and preprocessing steps.
    print(val_df.head())
