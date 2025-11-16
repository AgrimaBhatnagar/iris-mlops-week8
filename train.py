#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import mlflow
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report


def poison_data(X, y, frac, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    X = X.copy()
    n = X.shape[0]
    k = int(frac * n)
    idx = rng.choice(n, k, replace=False)
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    for i in idx:
        X[i, :] = rng.uniform(mins, maxs)
    return X, y, idx


def plot_cm(cm, labels, out):
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def main(args):
    mlflow.set_experiment(args.experiment_name)

    iris = load_iris()
    X, y = iris.data, iris.target
    labels = iris.target_names

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.25, random_state=args.seed, stratify=y
    )

    X_poison, y_poison, poisoned_idx = poison_data(
        X_train, y_train, args.poison_fraction, np.random.default_rng(args.seed)
    )

    with mlflow.start_run(run_name=f"poison-{args.poison_fraction}"):

        mlflow.log_param("poison_fraction", args.poison_fraction)

        model = RandomForestClassifier(n_estimators=100, random_state=args.seed)
        model.fit(X_poison, y_poison)

        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average="macro")

        mlflow.log_metric("val_accuracy", acc)
        mlflow.log_metric("val_f1", f1)

        cm = confusion_matrix(y_val, y_pred)
        plot_cm(cm, labels, "confusion.png")
        mlflow.log_artifact("confusion.png")

        with open("report.txt", "w") as f:
            f.write(classification_report(y_val, y_pred, target_names=labels))

        mlflow.log_artifact("report.txt")

        joblib.dump(model, "model.joblib")
        mlflow.log_artifact("model.joblib")

    print(f"Done: Poison={args.poison_fraction}, Accuracy={acc:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--poison_fraction", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--experiment_name", type=str, default="iris_poisoning")
    args = p.parse_args()
    main(args)
