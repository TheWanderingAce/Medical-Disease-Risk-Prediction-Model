from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, 
    precision_score,
    recall_score, 
    f1_score, 
    confusion_matrix
)
    

from src.preprocessing import split_data

print("model training start!")

def train_model():
    print("inside model training function")
    X_train, X_test, y_train, y_test = split_data()

    model = RandomForestClassifier(
        n_estimators = 200,
        max_depth = 15,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state = 45,
        n_jobs=-1
    )
    

    model.fit(X_train, y_train)
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)

    print("Training accuracy: ", train_accuracy)
    print("Testing accuracy: ", test_accuracy)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print("Accuracy:", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1-score ", f1)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


if __name__ == "__main__":

    train_model()
