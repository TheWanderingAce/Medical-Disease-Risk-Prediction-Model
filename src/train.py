from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from sklearn.utils.class_weight import compute_sample_weight

from src.preprocessing import split_data
from src.models.logistic_regression import create_model as create_logistic_model
from src.models.random_forest import create_model as create_random_forest_model
from src.models.gradient_boosting import create_model as create_gradient_boosting_model


def evaluate_model(model, X_val, y_val):
    # Make predictions
    y_pred = model.predict(X_val)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, zero_division=0)
    recall = recall_score(y_val, y_pred, zero_division=0)
    f1 = f1_score(y_val, y_pred, zero_division=0)

    # Display results
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_val, y_pred))

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def train_models():
    # Load and split dataset
    X_train, X_test, y_train, y_test = split_data()

    # Create validation set from training data
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=45,
        stratify=y_train
    )

    # Create models
    models = {
        "Logistic Regression": create_logistic_model(),
        "Random Forest": create_random_forest_model(),
        "Weighted Gradient Boosting": create_gradient_boosting_model(),
        # "Non-Weighted Gradient Boosting": create_gradient_boosting_model(),
    }

    results = {}

    # Train and evaluate each model
    for name, model in models.items():
        print(f"\n{'=' * 40}")
        print(f"Training {name}")
        print(f"{'=' * 40}")

        if name != "Weighted Gradient Boosting":
            model.fit(X_train, y_train)
        else:
            sample_weights = compute_sample_weight("balanced", y_train)
            model.fit(X_train, y_train, sample_weight=sample_weights)

        results[name] = evaluate_model(
            model,
            X_val,
            y_val
        )

    # Compare model performance
    print("\nMODEL COMPARISON")
    print("-" * 40)

    for name, metrics in results.items():
        print(
            f"{name}: "
            f"Accuracy={metrics['accuracy']:.4f}, "
            f"Precision={metrics['precision']:.4f}, "
            f"Recall={metrics['recall']:.4f}, "
            f"F1={metrics['f1']:.4f}"
        )

    return models, results


if __name__ == "__main__":
    train_models()