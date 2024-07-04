# load_data.py

import pandas as pd
from preprocess import preprocess_text  # Ensure preprocess.py is in the same directory

def load_datasets(train_path, test_path, val_path):
    # Load datasets
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    val_df = pd.read_csv(val_path)
    
    # Preprocess text
    train_df['cleaned_text'] = train_df['text'].apply(preprocess_text)
    test_df['cleaned_text'] = test_df['text'].apply(preprocess_text)
    val_df['cleaned_text'] = val_df['text'].apply(preprocess_text)
    
    return train_df, test_df, val_df

if __name__ == "__main__":
    train_df, test_df, val_df = load_datasets('emotion_dataset/training.csv', 'emotion_dataset/test.csv', 'emotion_dataset/validation.csv')
    print("Training Data:")
    print(train_df.head())
    print("\nTest Data:")
    print(test_df.head())
    print("\nValidation Data:")
    print(val_df.head())
