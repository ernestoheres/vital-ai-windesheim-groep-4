import sys
import os
sys.path.append(os.path.abspath("../../src"))

import optuna
import pandas as pd
import numpy as np
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

def export_prediction_set(test_patient_ids, df_original, y_pred):
    test_df = df_original[df_original['Patient_ID'].isin(test_patient_ids)].copy()
    test_df = test_df.sort_values(['Patient_ID', 'Hour'])

    # Ground truth
    true_labels = test_df[['Patient_ID', 'SepsisLabel']]
    true_labels.to_csv('testset (with label).csv', index=False)

    # Predictions 
    predictions = test_df[['Patient_ID']].copy()
    predictions['SepsisLabel'] = y_pred
    predictions.to_csv('predictions.csv', index=False)

def print_utiltiy_score(
        testset: str = "testset (with label).csv",
        predictions: str = "predictions.csv"
) -> None:
    utility = evaluate_sepsis_score(testset, predictions)
    print(utility)

def read_dataset() -> pd.DataFrame:
    return pd.read_csv('../../data/train_data.csv', sep=',')

def prep_dataset(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()

    # Onnodige kolommen verwijderen
    df = df.drop(columns=['Unnamed: 0'])

    # Sepsis_Future correct instellen
    df = df.sort_values(['Patient_ID', 'Hour'])

    df['Sepsis_Future'] = df.groupby('Patient_ID')['SepsisLabel'].shift(-6)
    df = df.dropna(subset=['Sepsis_Future'])
    
    return df


def train_test_split_by_patient(
    data: pd.DataFrame, 
    group_col: str = 'Patient_ID', 
    test_size=0.2, 
    random_state=42
):
    df = data.copy()
    patients = df[group_col].unique()

    train_patients, test_patients = train_test_split(
        patients, test_size=test_size, random_state=random_state
    )

    return train_patients, test_patients

TrainTestSplit = tuple[pd.DataFrame, pd.Series, np.array, pd.DataFrame, pd.Series, np.array]
def get_train_test_data_by_patient(
    data: pd.DataFrame,
    train_patients: pd.DataFrame, 
    test_patients: pd.DataFrame,
    y_target: str = 'Sepsis_Future',
    group_col: str = 'Patient_ID',
    delete_patient_ids: bool = False
) -> TrainTestSplit:
    df = data.copy()

    qsofa_features = [col for col in df.columns if 'qsofa' in col.lower()]

    sofa_features = [
        col for col in df.columns
        if 'sofa' in col.lower() and col not in qsofa_features
    ]

    if 'SF_ratio' in df.columns:
        sofa_features.append('SF_ratio')

    unit_features = [col for col in df.columns if 'unit' in col.lower()]

    drop_cols = [
        'SepsisLabel', 
        'Sepsis_Future',
        *qsofa_features, 
        *sofa_features, 
        *unit_features
    ]

    train = train_patients.copy()
    test = test_patients.copy()

    train_df = df[df[group_col].isin(train)]
    test_df = df[df[group_col].isin(test)]

    X_train = train_df.drop(columns=drop_cols)
    y_train = train_df[y_target].astype(int)

    X_test = test_df.drop(columns=drop_cols)
    y_test = test_df[y_target].astype(int)

    # Patient_ID is nodig voor het bereken van de utility score, maar het model mag niet trainen hierop
    # Daarom worden deze eruit gefilterd en daarna verwijderd uit de test sets
    train_patient_ids = X_train['Patient_ID'].values
    test_patient_ids = X_test['Patient_ID'].values

    if delete_patient_ids:
        X_train = X_train.drop(columns=['Patient_ID'])
        X_test = X_test.drop(columns=['Patient_ID'])

    return X_train, y_train, train_patient_ids, X_test, y_test, test_patient_ids

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

    score = f1_score(
        y_test,
        preds,
    )

    trial.report(score, step=0)

    if trial.should_prune():
        raise optuna.TrialPruned()

    return score

def run_all_experiments(
    train_patients,
    test_patients,
    n_trials=100,
    n_jobs=1
):

    all_results = []

    model_names = [
        "xgb",
        "lgbm",
        "catboost",
    ]

    for feature_set_name, feature_kwargs in FEATURE_CONFIGS.items():

        print("\n" + "=" * 80)
        print(f"FEATURE SET: {feature_set_name}")
        print("=" * 80)

        df = read_dataset()
        df = prep_dataset(df)
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
            delete_patient_ids=True,
        )

        print(f"\nTrain shape: {X_train.shape}")
        print(f"Test shape: {X_test.shape}")

        print(f"Train positives: {y_train.sum()}")
        print(f"Test positives: {y_test.sum()}")

        for model_name in model_names:

            print("\n" + "-" * 60)
            print(f"RUNNING MODEL: {model_name}")
            print("-" * 60)

            study = optuna.create_study(
                direction="maximize",

                pruner=optuna.pruners.MedianPruner(
                    n_startup_trials=10,
                    n_warmup_steps=5,
                ),

                sampler=optuna.samplers.TPESampler(
                    seed=42,
                ),
            )

            study.optimize(
                lambda trial: objective(
                    trial,
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    model_name,
                ),

                n_trials=n_trials,

                # parallel CPU workers
                n_jobs=n_jobs,

                show_progress_bar=True,
            )

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

            print("\nBEST RESULT")
            print(result)

    results_df = pd.DataFrame(all_results)

    results_df = results_df.sort_values(
        by="f1",
        ascending=False,
    )

    return results_df

if __name__ == "__main__":
    df = read_dataset()

    train_patients, test_patients = train_test_split_by_patient(df)

    results = run_all_experiments(
        train_patients=train_patients,
        test_patients=test_patients,
        n_trials=50,
    )

    print(results)