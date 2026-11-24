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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from src.data import download_from_gcs


def main():


    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    download_from_gcs(config["data"]["gcs_bucket"], config["data"]["gcs_cleaned_blob"], config["data"]["cleaned_path"])

    mlflow.set_experiment(config["mlflow"]["experiment_name"])

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

    with mlflow.start_run():
        pipeline.fit(X_train, y_train)
        y_val_prob = pipeline.predict_proba(X_val)[:,1]
        y_val_pred = (y_val_prob >= config["evaluation"]["threshold"]).astype(int)

        mlflow.log_params(config["model"]["params"])
        mlflow.log_params({
            "test_size": config["split"]["test_size"],
            "val_size": config["split"]["val_size"],
            })


        
        acc = accuracy_score(y_val, y_val_pred)
        prec = precision_score(y_val, y_val_pred)
        rec = recall_score(y_val, y_val_pred)
        f1 = f1_score(y_val, y_val_pred)
        f2 = fbeta_score(y_val, y_val_pred, beta=2)
        cm = confusion_matrix(y_val, y_val_pred)

        mlflow.log_metrics({
                "val_f2": f2,
                "val_recall": rec,
                "val_precision": prec,
                "val_f1": f1,
                "val_accuracy": acc,
                })

        print("Validation results")
        print(f"F2:        {f2:.3f}")
        print(f"Recall:    {rec:.3f}")
        print(f"Precision: {prec:.3f}")
        print(f"F1:        {f1:.3f}")
        print(f"Accuracy:  {acc:.3f}")
        print("Confusion matrix:")
        print(cm)

        fig, ax = plt.subplots()
        ConfusionMatrixDisplay(cm, display_labels=["No failure", "Failure"]).plot(ax=ax, colorbar=False)
        ax.set_title("Validation confusion matrix")
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)

        feature_names = [name.split("__")[1] for name in pipeline.named_steps["preprocessor"].get_feature_names_out()]
        importances = pd.Series(pipeline.named_steps["model"].feature_importances_, index=feature_names).sort_values()
        fig, ax = plt.subplots()
        importances.plot.barh(ax=ax)
        ax.set_title("Feature importance")
        ax.set_xlabel("Importance")
        fig.tight_layout()
        mlflow.log_figure(fig, "feature_importance.png")
        plt.close(fig)

        mlflow.sklearn.log_model(pipeline, name="model", input_example=X_train.head(5))
        mlflow.log_artifact("config.yaml")

        joblib.dump(pipeline, config["model"]["path"])
        print(f"Model saved to {config['model']['path']}")



if __name__ == "__main__":
    main()

