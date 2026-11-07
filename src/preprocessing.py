import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_loader import load_data

TARGET_COLUMN = "Diabetes_binary"

def split_data(test_size = 0.2, random_state = 45):

    '''
    Load the diabetes dataset and split it into
    training and testing sets.

    Returns:
        X_train: Training features
        X_test: Testing features
        y_train: Training labels
        y_test: Testing labels
    '''
    df = load_data()

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=45,
        stratify=y
    )
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":

    X_train, X_test, y_train, y_test = split_data()

    print("Training features:", X_train.shape)
    print("Testing features:", X_test.shape)

    print("Training labels:", y_train.shape)
    print("Testing labels:", y_test.shape)

    print("\nTraining class distribution:")
    print(y_train.value_counts(normalize=True))

    print("\nTesting class distribution:")
    print(y_test.value_counts(normalize=True))