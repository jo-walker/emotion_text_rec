import pandas as pd
from preprocess import preprocess_text

def load_datasets(train_path, test_path, val_path, single_text_path):
    """
    Loads the datasets from CSV files, preprocesses the text data, and returns the cleaned datasets.
    """
    # Load datasets
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    val_df = pd.read_csv(val_path)
    single_text_df = pd.read_csv(single_text_path, header=None, names=['text'])
    
    # Preprocess text
    train_df['cleaned_text'] = train_df['text'].apply(preprocess_text)
    test_df['cleaned_text'] = test_df['text'].apply(preprocess_text)
    val_df['cleaned_text'] = val_df['text'].apply(preprocess_text)
    single_text_df['cleaned_text'] = single_text_df['text'].apply(preprocess_text)
    
    return train_df, test_df, val_df, single_text_df

def load_unlabeled_text(unlabeled_text_path):
    unlabeled_df = pd.read_csv(unlabeled_text_path, header=None, names=['text'])
    return unlabeled_df

# Main execution
if __name__ == "__main__":
    train_df, test_df, val_df, single_text_df = load_datasets(
        'emotion_dataset/training.csv', 
        'emotion_dataset/test.csv', 
        'emotion_dataset/validation.csv',
        'emotion_dataset/unlabeled_text.csv'
    )
    print("Training Data:")
    print(train_df.head())
    print("\nTest Data:")
    print(test_df.head())
    print("\nValidation Data:")
    print(val_df.head())
    print("\nSingle Text Data:")
    print(single_text_df.head())
