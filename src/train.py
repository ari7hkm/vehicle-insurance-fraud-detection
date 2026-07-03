import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from data_loader import load_data
from preprocessing import preprocess
from config import *


def main():
    df = load_data(DATA_PATH)
    df = preprocess(df)
    print(df.shape)

    X = df.drop(columns=TARGET_COLUMN)
    y = df[TARGET_COLUMN]

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, 
                                                        random_state=RANDOM_STATE, stratify=y)
    
    smote = SMOTE(random_state=0)
    x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)

    classifier = RandomForestClassifier(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, min_samples_split=MIN_SAMPLES_SPLIT,
                                        min_samples_leaf=MIN_SAMPLES_LEAF, verbose=VERBOSE, random_state=RANDOM_STATE,
                                        bootstrap=True, max_features=MAX_FEATURES)

    model = classifier.fit(x_train_resampled, y_train_resampled)
    y_pred = model.predict(x_test)

    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"Precision: {precision_score(y_test, y_pred)}")
    print(f"Recall: {recall_score(y_test, y_pred)}")
    print(f"F1 Score: {f1_score(y_test, y_pred)}")

if __name__ == "__main__":
    main()