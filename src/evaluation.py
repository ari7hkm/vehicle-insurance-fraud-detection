from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)

    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "f1-score": f1_score(y_test, y_pred)
    }
