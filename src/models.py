from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from config import RANDOM_STATE, PARAM_DIST



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
