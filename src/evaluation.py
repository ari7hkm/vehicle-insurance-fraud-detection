import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pathlib import Path



def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)

    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "f1-score": f1_score(y_test, y_pred)
    }


def save_metrics(metrics, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)
        print(f"Metrics saved to {path}")
