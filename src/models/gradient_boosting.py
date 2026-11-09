from sklearn.ensemble import GradientBoostingClassifier


def create_model():
    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=3,
        min_samples_leaf=5,
        random_state=45
    )

    return model 