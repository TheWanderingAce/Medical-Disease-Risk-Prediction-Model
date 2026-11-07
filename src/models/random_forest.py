from sklearn.ensemble import RandomForestClassifier


def create_model():
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=45,
        n_jobs=-1
    )

    return model