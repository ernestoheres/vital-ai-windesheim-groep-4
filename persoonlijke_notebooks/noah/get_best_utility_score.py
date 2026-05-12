import re
import joblib
import pandas as pd
from pathlib import Path
from scepsis_prediction.feature_engineering import add_all_features
from scepsis_prediction.evaluation import evaluate_sepsis_score
from helpers.notebook_helpers import (
    read_dataset,
    prep_dataset,
    train_test_split_by_patient,
    get_train_test_data_by_patient,
    export_prediction_set
)

FEATURE_CONFIGS = {
    "all": {
        "include_rolling": True,
        "include_temporal": True,
    },
    "temporal_only": {
        "include_rolling": False,
        "include_temporal": True,
    },
    "rolling_only": {
        "include_rolling": True,
        "include_temporal": False,
    },
    "base": {
        "include_rolling": False,
        "include_temporal": False,
    },
}


def __combine_paths(*csv_paths) -> pd.DataFrame:

    dfs = []

    for path in csv_paths:
        df = pd.read_csv(path)

        # datum uit bestandsnaam halen
        filename = Path(path).stem

        match = re.search(
            r'(\d{2}-\d{2}-\d{4})',
            filename
        )

        run_date = (
            pd.to_datetime(
                match.group(1),
                format="%m-%d-%Y"
            )
            if match else pd.NaT
        )

        # toevoegen aan ALLE rows
        df["run_date"] = run_date

        dfs.append(df)

    combined_csv = pd.concat(
        dfs,
        ignore_index=True
    )

    return combined_csv



def __select_best_model(
        models_df: pd.DataFrame, 
        use_min_recall: bool,
    ):

    original_df = read_dataset()
    original_df = prep_dataset(original_df)

    models = models_df.copy()
    MIN_RECALL = 0.3

    if use_min_recall: 
        models = models[models['recall'] >= MIN_RECALL]
    else:
        models = models.nlargest(5, 'f1')

    predictions = {}

    for _, row in models.iterrows():
        feature_set = row["feature_set"]
        model_name = row["model"]
        run_date = row["run_date"]
    
        df_copy = original_df.copy()
        df_copy = add_all_features(
            df_copy,
            **FEATURE_CONFIGS[feature_set]
        )

        df_copy = df_copy.ffill()
        df_copy = df_copy.dropna()

        train_patients, test_patients = train_test_split_by_patient(df_copy)
        _, _, _, X_test, _, test_patient_ids = get_train_test_data_by_patient(
            df_copy,
            train_patients,
            test_patients,
            delete_patient_ids=True,
        )

        model_path = (
            f"optuna_storage/saved_models/"
            f"{feature_set}_{model_name}_"
            f"{run_date.strftime('%m-%d-%Y')}.pkl"
        )

        loaded = joblib.load(model_path)

        model = loaded["model"]
        threshold = loaded.get("threshold", 0.5)

        proba = model.predict_proba(X_test)[:, 1]
        y_pred = (proba >= threshold).astype(int)

        export_prediction_set(
            test_patient_ids,
            df_copy,
            y_pred
        )

        utility = evaluate_sepsis_score("testset (with label).csv", "predictions.csv")
        
        predictions[
            model_path
        ] = utility

    best_model = max(predictions, key=predictions.get)
    best_score = predictions[best_model]

    print(f"{best_model}: {best_score}")

    return best_model
        

    
def return_best_model(
         *csv_paths,
        use_min_recall: bool = False
    ):

    combined_csv = __combine_paths(*csv_paths)
    best_model = __select_best_model(combined_csv, use_min_recall)

    return best_model


if __name__ == "__main__":
    model = return_best_model("optuna_storage/results_10-05-2026.csv", "optuna_storage/results_11-05-2026.csv", use_min_recall=True)