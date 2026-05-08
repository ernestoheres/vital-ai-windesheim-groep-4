import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Scepsis voorspellen tot 6 uur voor diagnose

    In de eerste cyclus gaan we eerst een baseline maken, en kort itereren
    Hierna gaan we steeds meer features en andere functies toevoegen en kijken hoeveel de baseline veranderd
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    **Teamcontext**: ISALA x Windesheim

    Doel: een praktisch en uitlegbaar voorspelproces dat vroegtijdige signalering ondersteunt.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Aanpak in deze notebook

    1. Data inladen en snel controleren
    2. EDA op correlaties en datakwaliteit
    3. Modeltraining met patient-level split
    4. Threshold optimaliseren op utility
    5. Eindpredictions exporteren
    """)
    return


@app.cell
def _():
    import pandas as pd
    import marimo as mo
    from sklearn.model_selection import GridSearchCV, train_test_split as tts_fs
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split as tts_model
    from xgboost import XGBClassifier
    import matplotlib.pyplot as plt_cm
    import seaborn as sns_cm
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.pyplot as plt_ut
    from scepsis_prediction import evaluation as eval_utils


    return (
        XGBClassifier,
        confusion_matrix,
        eval_utils,
        mo,
        np,
        pd,
        plt,
        plt_cm,
        plt_ut,
        sns,
        sns_cm,
        tts_model,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1) Data inladen

    We laden train- en testdata in en doen daarna een snelle sanity check op vorm, kolommen en basisstatistiek.
    """)
    return


@app.cell
def _(pd):
    df = pd.read_csv("data/train_data.csv")
    df_test = pd.read_csv("data/test_data.csv")
    return df, df_test


@app.cell
def _(df):
    df.describe()
    return


@app.cell
def _(df, plt, sns):
    # compute correlation with target only
    corr = df.corr(numeric_only=True)['SepsisLabel'].sort_values(ascending=False)

    # optional: drop the label itself
    corr = corr.drop('SepsisLabel')

    plt.figure(figsize=(6, 10))

    sns.heatmap(
        corr.to_frame(),   # turn into 2D for heatmap
        annot=True,
        fmt=".2f",
        cmap='coolwarm',
        linewidths=0.5,
        cbar_kws={"shrink": 0.8}
    )

    plt.title("Correlation with SepsisLabel", fontsize=14)
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2) Verkenning

    Eerst kijken we naar correlaties en missende waarden om featurekeuzes bewuster te maken.
    """)
    return


@app.cell
def _(df):
    missingness = (
        df.isna()
        .mean()
        .sort_values(ascending=False)
        .rename("missing_ratio")
    )

    class_balance = df["SepsisLabel"].value_counts(normalize=True).rename("ratio")

    print("Top 10 missende features:")
    print(missingness.head(10))
    print("\nClass balance (SepsisLabel):")
    print(class_balance)

    missingness.head(15)
    return


@app.cell
def _():
    # data = df.select_dtypes(include="number").dropna(subset=["SepsisLabel"])
    # sample_n = min(100_000, len(data))
    # data_sample = data.sample(n=sample_n, random_state=42)

    # X_fs = data_sample.drop(columns=["SepsisLabel"])
    # y_fs = data_sample["SepsisLabel"]

    # X_train_fs, X_val_fs, y_train_fs, y_val_fs = tts_fs(
    #     X_fs, y_fs, test_size=0.2, random_state=42, stratify=y_fs
    # )

    # pipeline = Pipeline([
    #     ("imputer", SimpleImputer(strategy="median")),
    #     ("scaler", StandardScaler()),
    #     (
    #         "model",
    #         LogisticRegression(
    #             penalty="l1",
    #             solver="saga",
    #             max_iter=2000,
    #             random_state=42,
    #         ),
    #     ),
    # ])

    # grid = GridSearchCV(
    #     pipeline,
    #     param_grid={"model__C": [0.01, 0.1, 1]},
    #     cv=3,
    #     scoring="roc_auc",
    #     n_jobs=-1,
    #     verbose=0,
    # )

    # grid.fit(X_train_fs, y_train_fs)

    # coefs = pd.Series(
    #     grid.best_estimator_.named_steps["model"].coef_[0],
    #     index=X_fs.columns,
    # )
    # selected_features = coefs[coefs != 0].sort_values(key=lambda s: s.abs(), ascending=False)

    # print("Best params:", grid.best_params_)
    # print("Best CV ROC-AUC:", round(grid.best_score_, 4))
    # print("\nTop geselecteerde features:")
    # print(selected_features.head(15))

    # selected_features
    return


@app.cell
def _(df):
    base_features_raw = [
        'HR',
        'Hour',
        'O2Sat',
        'Temp',
        'SBP',
        'MAP',
        'DBP',
        'Resp',
        'EtCO2',
        'Platelets',
        'PaCO2',
        'Glucose',
        'BUN',
        'BaseExcess',
        'HCO3',
        'FiO2',
        'pH',
        'SaO2',
        'AST',
        'Alkalinephos',
        'Chloride',
        'Lactate',
        'Magnesium',
        'Phosphate',
        'Potassium',
        'Bilirubin_direct',
        'Bilirubin_total',
        'TroponinI',
        'WBC',
        'PTT',
        'Fibrinogen',
        'Calcium',
        'Creatinine',
        'Hct',
        'Hgb',
        'Age',
        'Gender',
        'HospAdmTime',
    ]

    rolling_stat_cols_raw = [
        'HR',
        'O2Sat',
        'Temp',
        'SBP',
        'MAP',
        'DBP',
        'Resp',
        'EtCO2',
        'FiO2',
        'SaO2',
        'Glucose',
        'Lactate',
        'WBC',
        'Creatinine',
        'Platelets',
        'BUN',
        'pH',
        'PaCO2',
        'HCO3',
        'BaseExcess',
    ]

    delta_cols_raw = [
        'HR',
        'O2Sat',
        'Temp',
        'SBP',
        'MAP',
        'Resp',
        'Glucose',
        'Lactate',
        'Creatinine',
        'WBC',
    ]

    rolling_windows = [3, 6]

    base_features = [f for f in base_features_raw if f in df.columns]
    rolling_stat_cols = [f for f in rolling_stat_cols_raw if f in df.columns]
    delta_cols = [f for f in delta_cols_raw if f in df.columns]

    base_features
    return base_features, delta_cols, rolling_stat_cols, rolling_windows


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3) Modeltraining (XGBoost)

    We splitsen op patientniveau (niet op losse rijen), vullen missende waarden met train-mediaan en trainen daarna een robuuste baseline.
    """)
    return


@app.cell
def _(
    XGBClassifier,
    base_features,
    delta_cols,
    df,
    df_test,
    rolling_stat_cols,
    rolling_windows,
    tts_model,
):


    target_col = "SepsisLabel"
    patient_col = "Patient_ID"

    # Werk op kopieen zodat de originele data intact blijft
    df_model = df.copy()
    df_test_model = df_test.copy()
    df_model.columns = df_model.columns.str.strip()
    df_test_model.columns = df_test_model.columns.str.strip()

    # Patient-level stratificatie: ooit sepsis vs nooit sepsis
    patient_level = (
        df_model.groupby(patient_col)[target_col]
        .max()
        .reset_index()
    )

    train_ids, val_ids = tts_model(
        patient_level[patient_col],
        test_size=0.2,
        random_state=42,
        stratify=patient_level[target_col],
    )

    train_df = df_model[df_model[patient_col].isin(train_ids)].copy()
    val_df = df_model[df_model[patient_col].isin(val_ids)].copy()

    def add_temporal_features(frame):
        frame = frame.sort_values([patient_col, "Hour"]).copy()

        if {"HR", "SBP"}.issubset(frame.columns):
            frame["ShockIndex"] = frame["HR"].div(frame["SBP"].replace(0, float("nan")))

        if {"SBP", "DBP"}.issubset(frame.columns):
            frame["PulsePressure"] = frame["SBP"] - frame["DBP"]

        temporal_feature_cols = []
        rolling_targets = list(rolling_stat_cols)
        for derived_col in ["ShockIndex", "PulsePressure"]:
            if derived_col in frame.columns:
                rolling_targets.append(derived_col)

        for col in rolling_targets:
            grouped = frame.groupby(patient_col)[col]

            for window in rolling_windows:
                feature_map = {
                    "mean": grouped.transform(
                        lambda s: s.rolling(window=window, min_periods=1).mean()
                    ),
                    "std": grouped.transform(
                        lambda s: s.rolling(window=window, min_periods=2).std()
                    ),
                    "min": grouped.transform(
                        lambda s: s.rolling(window=window, min_periods=1).min()
                    ),
                    "max": grouped.transform(
                        lambda s: s.rolling(window=window, min_periods=1).max()
                    ),
                }

                for stat_name, values in feature_map.items():
                    feature_name = f"{col}_roll{window}_{stat_name}"
                    frame[feature_name] = values
                    temporal_feature_cols.append(feature_name)

        for col in delta_cols:
            feature_name = f"{col}_delta_1"
            frame[feature_name] = frame.groupby(patient_col)[col].transform(lambda s: s.diff())
            temporal_feature_cols.append(feature_name)

        if "ShockIndex" in frame.columns:
            temporal_feature_cols.append("ShockIndex")
        if "PulsePressure" in frame.columns:
            temporal_feature_cols.append("PulsePressure")

        return frame, temporal_feature_cols

    train_df, temporal_feature_cols = add_temporal_features(train_df)
    val_df, _ = add_temporal_features(val_df)
    df_test_model, _ = add_temporal_features(df_test_model)

    model_features = base_features + temporal_feature_cols

    X_train = train_df[model_features].copy()
    y_train = train_df[target_col].copy()
    X_val = val_df[model_features].copy()
    y_val = val_df[target_col].copy()
    X_test = df_test_model[model_features].copy()

    model = XGBClassifier(
        random_state=42,
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        scale_pos_weight=39,
        tree_method="hist",
        eval_metric="logloss",
        n_jobs=-1,
    )

    fitted_model = model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    y_val_proba = model.predict_proba(X_val)[:, 1]
    model
    return (
        X_test,
        df_test_model,
        model,
        patient_col,
        target_col,
        val_df,
        y_val,
        y_val_proba,
    )


@app.cell
def _(best_threshold, confusion_matrix, plt_cm, sns_cm, y_val, y_val_proba):
    y_val_pred = (y_val_proba >= best_threshold).astype(int)
    cm = confusion_matrix(y_val, y_val_pred)

    plt_cm.figure(figsize=(6, 4))
    sns_cm.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Pred 0", "Pred 1"],
        yticklabels=["Actual 0", "Actual 1"],
    )
    plt_cm.title("Confusion Matrix @ best threshold")
    plt_cm.xlabel("Predicted")
    plt_cm.ylabel("Actual")
    plt_cm.tight_layout()
    plt_cm.show()

    y_val_pred
    return


@app.cell
def _(y_val, y_val_proba):
    from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score
    import matplotlib.pyplot as plt_pr

    precision, recall, _ = precision_recall_curve(y_val, y_val_proba)
    pr_auc = average_precision_score(y_val, y_val_proba)
    roc_auc = roc_auc_score(y_val, y_val_proba)

    plt_pr.figure(figsize=(6, 4))
    plt_pr.plot(recall, precision, label=f"PR-AUC = {pr_auc:.4f}")
    plt_pr.xlabel("Recall")
    plt_pr.ylabel("Precision")
    plt_pr.title("Precision-Recall Curve")
    plt_pr.legend(loc="lower left")
    plt_pr.tight_layout()
    plt_pr.show()

    print("PR-AUC:", round(float(pr_auc), 4))
    print("ROC-AUC:", round(float(roc_auc), 4))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 4) Thresholdkeuze

    We kiezen de classificatiedrempel niet op standaard 0.5, maar op maximale klinische utility op de validatieset.
    """)
    return


@app.cell
def _(eval_utils, np, patient_col, plt_ut, target_col, val_df, y_val_proba):

    val_eval_df = val_df[[patient_col, target_col]].copy().reset_index(drop=True)
    val_eval_df["proba"] = y_val_proba

    val_labels_path = "data/validation_labels.csv"
    val_predictions_path = "data/validation_predictions.csv"
    val_eval_df[[patient_col, target_col]].to_csv(val_labels_path, index=False)

    thresholds = np.arange(0.20, 1.00, 0.20)
    utilities = []

    for threshold in thresholds:
        pred_df = val_eval_df[[patient_col]].copy()
        pred_df["SepsisLabel"] = (val_eval_df["proba"] >= threshold).astype(int)
        pred_df.to_csv(val_predictions_path, index=False)
        score_at_threshold = eval_utils.evaluate_sepsis_score(
            label_csv=val_labels_path,
            prediction_csv=val_predictions_path,
        )
        utilities.append(score_at_threshold)

    best_idx = int(np.argmax(utilities))
    best_threshold = float(thresholds[best_idx])
    best_utility = float(utilities[best_idx])

    print(f"Best threshold: {best_threshold:.2f}")
    print(f"Best utility: {best_utility:.6f}")

    plt_ut.figure(figsize=(10, 5))
    plt_ut.plot(thresholds, utilities)
    plt_ut.axvline(best_threshold, linestyle="--")
    plt_ut.xlabel("Threshold")
    plt_ut.ylabel("Validation utility")
    plt_ut.title("Validation utility vs threshold")
    plt_ut.show()

    (
        best_threshold,
        best_utility,
        thresholds,
        utilities,
        val_eval_df,
        val_labels_path,
        val_predictions_path,
    )
    return (best_threshold,)


@app.cell
def _(X_test, best_threshold, df_test_model, model, patient_col):
    test_proba = model.predict_proba(X_test)[:, 1]
    test_preds = (test_proba >= best_threshold).astype(int)

    submission_df = df_test_model[[patient_col]].copy()
    submission_df["SepsisLabel"] = test_preds
    submission_df.to_csv("predictions.csv", index=False)

    print("Best threshold gebruikt:", round(float(best_threshold), 2))
    print("Predictions opgeslagen in predictions.csv")
    submission_df.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 5) Reflectie & bron

    - We evalueren niet alleen met ROC/PR, maar kiezen de threshold op utility (klinische kosten-baten).
    - Bron over stratified validatie: https://www.geeksforgeeks.org/machine-learning/stratified-k-fold-cross-validation
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Business Understanding
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Data understanding
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Crisp DM Cyclus 2
    """)
    return


app._unparsable_cell(
    r"""
    In deze cyclus wil ik kijken naar temporale features, hoe een patient progressed met scepsis. Daarnaast wil ik meer focussen op de klas imbalance
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Business understanding
    """)
    return


app._unparsable_cell(
    r"""
    schock index over tijd
    Qsofa en partial sofa over tijd

    gemiddeldes
    slopes
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data understanding
    """)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
