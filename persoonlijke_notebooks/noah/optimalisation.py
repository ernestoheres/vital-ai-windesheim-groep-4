import sys
import os
sys.path.append(os.path.abspath("../../src"))

import optuna
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from pathlib import Path
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from scepsis_prediction.feature_engineering import add_all_features
from sklearn.model_selection import train_test_split
from scepsis_prediction.evaluation import evaluate_sepsis_score
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from helpers.notebook_helpers import (
    export_prediction_set,
    print_utiltiy_score,
    read_dataset,
    prep_dataset,
    train_test_split_by_patient,
    get_train_test_data_by_patient,
)

def get_timestamp():
    return datetime.now().strftime("%d-%m-%Y")

def get_study_name(feature_set_name, model_name, trials):
    return f"{feature_set_name}_{model_name}_{trials}_{time_stamp}"

def create_model(trial, model_name):
    if model_name == "xgb":
        params = {
            "n_estimators": trial.suggest_int(
                "n_estimators",
                200,
                1500,
            ),

            "learning_rate": trial.suggest_float(
                "learning_rate",
                0.005,
                0.2,
                log=True,
            ),

            "max_depth": trial.suggest_int(
                "max_depth",
                3,
                12,
            ),

            "subsample": trial.suggest_float(
                "subsample",
                0.5,
                1.0,
            ),

            "colsample_bytree": trial.suggest_float(
                "colsample_bytree",
                0.5,
                1.0,
            ),

            "min_child_weight": trial.suggest_int(
                "min_child_weight",
                1,
                10,
            ),

            "gamma": trial.suggest_float(
                "gamma",
                0.0,
                5.0,
            ),

            "reg_alpha": trial.suggest_float(
                "reg_alpha",
                0.0,
                5.0,
            ),

            "reg_lambda": trial.suggest_float(
                "reg_lambda",
                0.0,
                5.0,
            ),

            # GPU
            "tree_method": "hist",
            "device": "cuda",

            "random_state": 42,
            "eval_metric": "logloss",
            "verbosity": 0,
        }

        model = XGBClassifier(**params)

    elif model_name == "lgbm":
        params = {
            "n_estimators": trial.suggest_int(
                "n_estimators",
                200,
                1500,
            ),

            "learning_rate": trial.suggest_float(
                "learning_rate",
                0.005,
                0.2,
                log=True,
            ),

            "max_depth": trial.suggest_int(
                "max_depth",
                3,
                12,
            ),

            "num_leaves": trial.suggest_int(
                "num_leaves",
                15,
                255,
            ),

            "subsample": trial.suggest_float(
                "subsample",
                0.5,
                1.0,
            ),

            "colsample_bytree": trial.suggest_float(
                "colsample_bytree",
                0.5,
                1.0,
            ),

            "min_child_samples": trial.suggest_int(
                "min_child_samples",
                5,
                100,
            ),

            "reg_alpha": trial.suggest_float(
                "reg_alpha",
                0.0,
                5.0,
            ),

            "reg_lambda": trial.suggest_float(
                "reg_lambda",
                0.0,
                5.0,
            ),
            "random_state": 42,
            "verbosity": -1,
        }

        model = LGBMClassifier(**params)

    elif model_name == "catboost":
        params = {
            "iterations": trial.suggest_int(
                "iterations",
                200,
                1500,
            ),

            "learning_rate": trial.suggest_float(
                "learning_rate",
                0.005,
                0.2,
                log=True,
            ),

            "depth": trial.suggest_int(
                "depth",
                4,
                12,
            ),

            "l2_leaf_reg": trial.suggest_float(
                "l2_leaf_reg",
                1.0,
                10.0,
            ),

            "random_strength": trial.suggest_float(
                "random_strength",
                0.0,
                5.0,
            ),

            "bagging_temperature": trial.suggest_float(
                "bagging_temperature",
                0.0,
                5.0,
            ),
            "random_state": 42,
            "verbose": 0,
        }

        model = CatBoostClassifier(**params)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model

def objective(
    trial,
    X_train,
    y_train,
    X_test,
    y_test,
    test_patient_ids,
    model_name,
):
    threshold = trial.suggest_float(
        "threshold",
        0.05,
        0.95,
    )

    model = create_model(
        trial,
        model_name,
    )

    if model_name == "catboost":
        model.fit(
            X_train,
            y_train,
            eval_set=(X_test, y_test),
            use_best_model=True,
        )

    else:
        model.fit(
            X_train,
            y_train,
        )

    proba = model.predict_proba(X_test)[:, 1]

    preds = (
        proba >= threshold
    ).astype(int)

    temp_true = pd.DataFrame({
        "Patient_ID": test_patient_ids,
        "SepsisLabel": y_test.values,
    })

    temp_pred = pd.DataFrame({
        "Patient_ID": test_patient_ids,
        "SepsisLabel": preds,
    })

    temp_true_path = "temp_true.csv"
    temp_pred_path = "temp_pred.csv"

    temp_true.to_csv(
        temp_true_path,
        index=False,
    )

    temp_pred.to_csv(
        temp_pred_path,
        index=False,
    )

    score = evaluate_sepsis_score(
        temp_true_path,
        temp_pred_path,
    )

    trial.report(score, step=0)

    if trial.should_prune():
        raise optuna.TrialPruned()

    return score

def run_all_experiments(
    train_patients,
    test_patients,
    model_names_run,
    feature_configs_run,
    n_trials=100,
    n_jobs=1
):

    all_results = []

    for feature_set_name, feature_kwargs in feature_configs_run.items():

        print("\n" + "=" * 80)
        print(f"FEATURE SET: {feature_set_name}")
        print("=" * 80)

        df = read_dataset()
        df = prep_dataset(df, False)
        df = add_all_features(
            df,
            **feature_kwargs,
        )

        df = df.ffill()
        df = df.dropna()

        (
            X_train,
            y_train,
            train_patient_ids,
            X_test,
            y_test,
            test_patient_ids,
        ) = get_train_test_data_by_patient(
            df,
            train_patients,
            test_patients,
            y_target='SepsisLabel',
            delete_patient_ids=True,
        )

        print(f"\nTrain shape: {X_train.shape}")
        print(f"Test shape: {X_test.shape}")

        print(f"Train positives: {y_train.sum()}")
        print(f"Test positives: {y_test.sum()}")

        for model_name in model_names_run:

            print("\n" + "-" * 60)
            print(f"RUNNING MODEL: {model_name}")
            print("-" * 60)

            study_name = get_study_name(
                feature_set_name,
                model_name,
                n_trials
            )

            study = optuna.create_study(
                study_name=study_name,
                storage=f"sqlite:///{OPTUNA_DB}",
                load_if_exists=True,

                direction="maximize",

                pruner=optuna.pruners.MedianPruner(
                    n_startup_trials=10,
                    n_warmup_steps=5,
                ),

                sampler=optuna.samplers.TPESampler(
                    seed=42,
                ),
            )
            
            try:
                study.optimize(
                    lambda trial: objective(
                        trial,
                        X_train,
                        y_train,
                        X_test,
                        y_test,
                        test_patient_ids,
                        model_name,
                    ),

                    n_trials=n_trials,

                    # parallel CPU workers
                    n_jobs=n_jobs,

                    show_progress_bar=True,
                )
            except Exception as e:
                print(f"ERROR IN {study_name}: {e}")
                continue  

            best_params = study.best_params

            best_threshold = best_params.pop(
                "threshold"
            )

            final_model = create_model(
                optuna.trial.FixedTrial(best_params),
                model_name,
            )

            if model_name == "catboost":
                final_model.fit(
                    X_train,
                    y_train,
                    eval_set=(X_test, y_test),
                    use_best_model=True,
                )

            else:
                final_model.fit(
                    X_train,
                    y_train,
                )


            final_proba = final_model.predict_proba(
                X_test
            )[:, 1]

            final_preds = (
                final_proba >= best_threshold
            ).astype(int)

            final_f1 = f1_score(
                y_test,
                final_preds,
            )

            final_precision = precision_score(
                y_test,
                final_preds,
                zero_division=0,
            )

            final_recall = recall_score(
                y_test,
                final_preds,
                zero_division=0,
            )

            final_auc = roc_auc_score(
                y_test,
                final_proba,
            )

            result = {
                "feature_set": feature_set_name,
                "model": model_name,

                "f1": final_f1,
                "precision": final_precision,
                "recall": final_recall,
                "roc_auc": final_auc,

                "threshold": best_threshold,

                "best_params": best_params,
            }

            all_results.append(result)

            results_df_temp = pd.DataFrame(all_results)
            results_df_temp.to_csv(
                RESULTS_CSV,
                index=False,
            )


            model_path = (
                MODELS_DIR /
                f"{feature_set_name}_{model_name}_{time_stamp}.pkl"
            )

            joblib.dump(
                {
                    "model": final_model,
                    "threshold": best_threshold,
                    "params": best_params,
                },
                model_path,
            )
            print("\nBEST RESULT")
            print(result)

    results_df = pd.DataFrame(all_results)

    results_df = results_df.sort_values(
        by="f1",
        ascending=False,
    )

    return results_df

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

model_names_first_test = [
    "xgb",
    "lgbm",
    "catboost",
]

model_names_second_test = [
    "lgbm",
    "catboost",
]

model_names_third_test = [
    "xgb",
    "lgbm",
]

FEATURE_CONFIG_SECOND_RUN = {
    "all": {
        "include_rolling": True,
        "include_temporal": True,
    },
    "rolling_only": {
        "include_rolling": True,
        "include_temporal": False,
    }
}

FEATURE_CONFIG_THIRD_RUN = FEATURE_CONFIG_SECOND_RUN


BASE_DIR = Path("./optuna_storage")
BASE_DIR.mkdir(exist_ok=True)

OPTUNA_DB = BASE_DIR / "sepsis_optuna.db"
RESULTS_CSV = BASE_DIR / "results.csv"
MODELS_DIR = BASE_DIR / "saved_models"

MODELS_DIR.mkdir(exist_ok=True)

time_stamp: str

if __name__ == "__main__":
    time_stamp = get_timestamp()

    df = read_dataset()

    train_patients, test_patients = train_test_split_by_patient(df)

    results = run_all_experiments(
        train_patients=train_patients,
        test_patients=test_patients,
        model_names_run=model_names_third_test,
        feature_configs_run=FEATURE_CONFIG_THIRD_RUN,
        n_trials=80,
    )

    print(results)

    try:
        filename = f"all_result_{time_stamp}"
        results.to_csv(filename, sep=',', index=False) 
    except Exception as e:
        print(f'Error while saving: {e}')