import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (confusion_matrix, roc_curve, auc, 
                             precision_recall_curve, average_precision_score)



def generate_plots(model, x_test, y_test, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    y_prob = model.predict_proba(x_test)[:, 1]
    y_pred = model.predict(x_test)

    plot_confusion_matrix(y_test, y_pred, save_dir)
    plot_roc_curve(y_test, y_prob, save_dir)
    plot_precision_recall_curve(y_test, y_prob, save_dir)
    plot_feature_importance(model, x_test, save_dir)
    


def plot_confusion_matrix(y_test, y_pred, save_dir):
    save_dir = Path(save_dir)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt='d', cbar=True, xticklabels=["Not Fraud", "Fraud"],
                yticklabels=["Not Fraud", "Fraud"], cmap="YlGnBu", annot_kws={"size": 14, "weight": "bold"},
                cbar_kws={"label": "Count"})
    
    plt.title("Confusion Matrix", fontsize=18, pad=20)
    plt.xlabel("Predicted Label", fontsize=14)
    plt.ylabel("True Label", fontsize=14)

    plt.tight_layout()
    plt.savefig(save_dir / "confusion_matrix.png", dpi=300)
    plt.close()


def plot_roc_curve(y_test, y_prob, save_dir):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 8))

    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC Curve (AUC={roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle='--')

    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("ROC Curve - Random Forest Classifier")

    plt.legend()
    plt.grid(True)

    plt.savefig(save_dir / "roc_curve.png", dpi=300)
    plt.close()


def plot_precision_recall_curve(y_test, y_prob, save_dir):
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)

    plt.figure(figsize=(8, 8))

    plt.plot(recall, precision, color="darkorange", lw=2, label=f"PR Curve (AP={ap:.2f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve - Random Forest Classifier")

    plt.legend()
    plt.grid(True)

    plt.savefig(save_dir / "PR_curve.png", dpi=300)
    plt.close()


def plot_feature_importance(model, x_test, save_dir):
    importances = model.feature_importances_
    feature_names = x_test.columns
    indices = np.argsort(importances)[::-1]

    k = 15
    top_indices = indices[:k]

    plt.figure(figsize=(10, 7))

    bars = plt.barh(feature_names[top_indices][::-1], importances[top_indices][::-1], height=0.55)
    bars[-1].set_color('#1f77b4')

    for bar in bars[:-1]:
        bar.set_color('#87ceeb')

    plt.xlabel("Feature Importance", fontsize=13)
    plt.title(f"Top {k} Most Important Features - Random Forest", fontsize=16, pad=15)

    plt.grid(axis='x', linestyle='--', alpha=0.4)
    plt.tight_layout()

    plt.savefig(save_dir / 'feature_importance.png', dpi=300)
    plt.close()
