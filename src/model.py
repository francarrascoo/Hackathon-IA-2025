import os
import joblib
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


ARTIFACT_PATH = os.path.join("artifacts", "biobio_rf_pipeline_v1.joblib")
TRAIN_CSV = os.path.join("data", "train_dataset_full.csv")


def load_data(path=TRAIN_CSV):
    df = pd.read_csv(path)
    return df


def build_pipeline(n_estimators=200, random_state=42):
    # feature groups
    numeric_features = ["Shape__Length", "hora_del_dia", "dia_semana", "mes"]
    categorical_features = ["Comuna", "type"]

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False)),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ])

    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", clf),
    ])

    return pipeline


def train_and_persist(df, pipeline, artifact_path=ARTIFACT_PATH, test_size=0.2, random_state=42):
    # Expect target column named 'target'
    if "target" not in df.columns:
        raise ValueError("Input dataframe must contain 'target' column")

    X = df.drop(columns=["target"])
    y = df["target"]

    # Quick check: if only one class present, abort gracefully
    unique = np.unique(y)
    if len(unique) < 2:
        print(f"Only one class present in target: {unique}. Aborting training.")
        return None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print("Fitting pipeline...")
    pipeline.fit(X_train, y_train)

    print("Evaluating on test set...")
    preds = pipeline.predict(X_test)
    probs = None
    try:
        probs = pipeline.predict_proba(X_test)[:, 1]
    except Exception:
        pass

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds, zero_division=0),
    }
    if probs is not None and len(np.unique(y_test)) == 2:
        try:
            metrics["roc_auc"] = roc_auc_score(y_test, probs)
        except Exception:
            metrics["roc_auc"] = None

    print("Metrics:")
    for k, v in metrics.items():
        print(f" - {k}: {v}")

    # Persist pipeline
    os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
    joblib.dump(pipeline, artifact_path)
    print(f"Saved pipeline to {artifact_path}")

    # Try to extract feature importances (map to feature names)
    try:
        preprocessor = pipeline.named_steps["preprocessor"]
        clf = pipeline.named_steps["clf"]

        # get feature names after transformation
        num_feats = preprocessor.transformers_[0][2]
        cat_pipeline = preprocessor.transformers_[1][1]
        cat_feats = []
        try:
            # sklearn >=1.0
            cat_feats = list(cat_pipeline.named_steps["onehot"].get_feature_names_out(preprocessor.transformers_[1][2]))
        except Exception:
            # fallback: create placeholder names
            cat_feats = [f"{c}_?" for c in preprocessor.transformers_[1][2]]

        feature_names = list(num_feats) + cat_feats

        importances = clf.feature_importances_
        feat_imp = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
        print("Top feature importances:")
        for name, imp in feat_imp[:20]:
            print(f" - {name}: {imp:.4f}")
    except Exception:
        print("Could not extract feature importances")

    return artifact_path, metrics


def main():
    print("Loading data from:", TRAIN_CSV)
    df = load_data()
    print("Loaded rows:", len(df))

    pipeline = build_pipeline()
    res = train_and_persist(df, pipeline)
    if res is None:
        print("Training did not complete (likely only one class present).")
    else:
        path, metrics = res
        print("Training finished.")


if __name__ == "__main__":
    main()
