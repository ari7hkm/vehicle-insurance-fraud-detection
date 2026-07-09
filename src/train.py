from sklearn.model_selection import train_test_split
from data_loader import load_data
from preprocessing import preprocess
from models import balance_dataset, tune_random_forest, save_model
from evaluation import evaluate_model, save_metrics
from visualization import generate_plots
from config import (DATA_PATH, TARGET_COLUMN, RANDOM_STATE, 
                    TEST_SIZE, MODELS_PATH, METRICS_PATH, PLOTS_PATH)


def main():
    df = prepare_dataset(DATA_PATH)

    x_train, x_test, y_train, y_test = split_data(df)
    
    x_train, y_train = balance_dataset(x_train, y_train)

    best_model = tune_random_forest(x_train, y_train)
    
    metrics = evaluate_model(best_model, x_test, y_test)
    
    for metric, value in metrics.items():
        print(f"{metric:<12}: {value:.4f}")

    save_model(best_model, MODELS_PATH)
    save_metrics(metrics, METRICS_PATH)
    generate_plots(best_model, x_test, y_test, PLOTS_PATH)


def prepare_dataset(data_path):
    df = load_data(data_path)
    return preprocess(df)


def split_data(data):
    X = data.drop(columns=TARGET_COLUMN)
    y = data[TARGET_COLUMN]

    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)


if __name__ == "__main__":
    main()
