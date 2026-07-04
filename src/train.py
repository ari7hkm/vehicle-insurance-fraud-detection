from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from data_loader import load_data
from preprocessing import preprocess
from config import *


def main():
    df = prepare_dataset(DATA_PATH)

    x_train, x_test, y_train, y_test = split_data(df)
    
    x_train, y_train = balance_dataset(x_train, y_train)

    best_model = tune_random_forest(x_train, y_train)
    
    metrics = evaluate_model(best_model, x_test, y_test)
    
    for metric, value in metrics.items():
        print(f"{metric:<12}: {value:.4f}")



def prepare_dataset(data_path):
    df = load_data(data_path)
    return preprocess(df)


def split_data(data):
    X = data.drop(columns=TARGET_COLUMN)
    y = data[TARGET_COLUMN]

    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)


def balance_dataset(x_train, y_train):
    smote = SMOTE(random_state=RANDOM_STATE)

    return smote.fit_resample(x_train, y_train)

def tune_random_forest(x, y):
    rf = RandomForestClassifier(random_state=RANDOM_STATE)
    random_search = RandomizedSearchCV(estimator=rf, param_distributions=PARAM_DIST, n_iter=10, cv=3, scoring="f1", 
                                       verbose=2, random_state=RANDOM_STATE, n_jobs=1)
    random_search.fit(x, y)
    print(f"Best parameters: {random_search.best_params_}")
    print(f"Best score: {random_search.best_score_}")

    return random_search.best_estimator_


def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)

    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "f1-score": f1_score(y_test, y_pred)
    }



if __name__ == "__main__":
    main()
