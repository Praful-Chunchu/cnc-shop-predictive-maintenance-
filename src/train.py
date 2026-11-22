import yaml
import joblib
import pandas as pd
from src.features import add_engineered_features
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)


def main():
    with open("config.yaml") as f:
        config = yaml.safe_load(f)


    raw_df = pd.read_csv(config["data"]["cleaned_path"])

    predictive_df = add_engineered_features(raw_df)

    X = predictive_df[config["features"]["numeric"] + config["features"]["categorical"]]
    y = predictive_df[config["features"]["target"]]


    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=config["split"]["test_size"], random_state=config["split"]["random_state"], stratify=y if config["split"]["stratify"] else None
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=config["split"]["val_size"] / (1-config["split"]["test_size"]), random_state=config["split"]["random_state"], stratify=y_temp if config["split"]["stratify"] else None
    )

    preprocessor = ColumnTransformer(
        transformers = [
            ('cat', OrdinalEncoder(categories=[config["features"]["type_order"]]), config["features"]["categorical"]),
            ("num", "passthrough", config["features"]["numeric"])

        ]
    )


    pipeline = Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('model', RandomForestClassifier(**config["model"]["params"], random_state=config["split"]["random_state"]))
        ]
    )


    pipeline.fit(X_train, y_train)
    y_val_pred = pipeline.predict(X_val)


    acc = accuracy_score(y_val, y_val_pred)
    prec = precision_score(y_val, y_val_pred)
    rec = recall_score(y_val, y_val_pred)
    f1 = f1_score(y_val, y_val_pred)
    f2 = fbeta_score(y_val, y_val_pred, beta=2)
    cm = confusion_matrix(y_val, y_val_pred)

    print("Validation results")
    print(f"F2:        {f2:.3f}")
    print(f"Recall:    {rec:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"F1:        {f1:.3f}")
    print(f"Accuracy:  {acc:.3f}")
    print("Confusion matrix:")
    print(cm)

    joblib.dump(pipeline, config["model"]["path"])
    print(f"Model saved to {config['model']['path']}")


if __name__ == "__main__":
    main()

