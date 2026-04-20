import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd
    import marimo as mo

    return mo, pd


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
def _(df):
    import seaborn as sns
    import matplotlib.pyplot as plt

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
    mo.md(r"""

    """)
    return


@app.cell
def _(df):
    def _():
        import seaborn as sns
        import matplotlib.pyplot as plt

        # compute correlation with target only
        corr1 = df.corr(numeric_only=True)['Unnamed: 0'].sort_values(ascending=False)

        # optional: drop the label itself
        corr1 = corr1.drop('Unnamed: 0')

        plt.figure(figsize=(6, 10))

        sns.heatmap(
            corr1.to_frame(),   # turn into 2D for heatmap
            annot=True,
            fmt=".2f",
            cmap='coolwarm',
            linewidths=0.5,
            cbar_kws={"shrink": 0.8}
        )

        plt.title("Correlation with SepsisLabel", fontsize=14)
        plt.yticks(rotation=0)

        plt.tight_layout()
        return plt.show()


    _()
    return


@app.cell
def _(df):
    def _():
        import pandas as pd
        from sklearn.model_selection import GridSearchCV, train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression

        # keep only numeric features + target
        data = df.select_dtypes(include='number').dropna(subset=['SepsisLabel'])

        # random sample
        data_sample = data.sample(n=100_000, random_state=42)

        X = data_sample.drop(columns=['SepsisLabel'])
        y = data_sample['SepsisLabel']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                penalty="l1",
                solver="saga",
                max_iter=2000,
                random_state=42
            ))
        ])

        param_grid = {
            "model__C": [0.01, 0.1, 1]
        }

        grid = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            cv=3,
            scoring="roc_auc",
            n_jobs=-1,
            verbose=1
        )

        grid.fit(X_train, y_train)

        print("Best params:", grid.best_params_)
        print("Best CV score:", grid.best_score_)

        best_model = grid.best_estimator_

        coefs = pd.Series(
            best_model.named_steps["model"].coef_[0],
            index=X.columns
        ).sort_values()

        print("\nAll coefficients:")
        print(coefs)

        print("\nSelected features:")
        print(coefs[coefs != 0].sort_values())

    _()
    return


@app.cell
def _():
    features_raw = [
        "Calcium",
        "Hgb",
        "EtCO2",
        "BUN"
    ]
    return (features_raw,)


@app.cell
def _(df, df_test, features_raw):
    from xgboost import XGBClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

    target_col = "SepsisLabel"
    patient_col = "Patient_ID"

    # Clean column names just in case
    df.columns = df.columns.str.strip()
    df_test.columns = df_test.columns.str.strip()

    # optional: also strip feature names if needed
    features = [f.strip() for f in features_raw]

    # patient-level labels for stratification:
    # 1 if patient ever becomes septic, else 0
    patient_level = (
        df.groupby(patient_col)[target_col]
        .max()
        .reset_index()
    )

    train_ids, val_ids = train_test_split(
        patient_level[patient_col],
        test_size=0.2,
        random_state=42,
        stratify=patient_level[target_col]
    )

    # split full rows by patient
    train_df = df[df[patient_col].isin(train_ids)].copy()
    val_df = df[df[patient_col].isin(val_ids)].copy()

    # keep only model features
    X_train = train_df[features].copy()
    y_train = train_df[target_col].copy()

    X_val = val_df[features].copy()
    y_val = val_df[target_col].copy()

    # fill missing values using only train stats
    medians = X_train.median()
    X_train = X_train.fillna(medians)
    X_val = X_val.fillna(medians)

    # hospital test set: no labels
    X_test = df_test[features].copy()
    X_test = X_test.fillna(medians)

    model = XGBClassifier(
        random_state=42,
        scale_pos_weight=39,
        device="cuda",
        max_depth=6,
        min_child_weight=10,
        subsample=0.8,
        colsample_bytree=0.8,
        n_estimators=500,
        learning_rate=0.05,
        reg_alpha=0.1,
        reg_lambda=1.0,
        gamma=0.1,
        tree_method="hist",
        eval_metric="aucpr",
        early_stopping_rounds=50,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=50
    )

    # local validation
    y_val_pred = model.predict(X_val)

    print("Accuracy:", accuracy_score(y_val, y_val_pred))
    print(confusion_matrix(y_val, y_val_pred))
    print(classification_report(y_val, y_val_pred))

    # final hospital predictions
    test_preds = model.predict(X_test)
    return (
        X_val,
        model,
        patient_col,
        target_col,
        test_preds,
        val_df,
        y_val,
        y_val_pred,
    )


@app.cell
def _(y_val, y_val_pred):
    def _():
        import matplotlib.pyplot as plt
        import seaborn as sns

        from sklearn.metrics import (
            confusion_matrix,
            roc_curve,
            roc_auc_score
        )

        # --- Confusion matrix ---
        cm = confusion_matrix(y_val, y_val_pred)

        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Pred 0", "Pred 1"],
                    yticklabels=["Actual 0", "Actual 1"])
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        return plt.show()


    _()
    return


@app.cell
def _(X_val, model, y_val):
    def _():
        from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score
        import matplotlib.pyplot as plt

        y_proba = model.predict_proba(X_val)[:, 1]  # ← X_val, not X_test

        precision, recall, thresholds = precision_recall_curve(y_val, y_proba)
        pr_auc = average_precision_score(y_val, y_proba)
        roc_auc = roc_auc_score(y_val, y_proba)

        plt.figure(figsize=(6, 4))
        plt.plot(recall, precision, label=f"PR-AUC = {pr_auc:.4f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend(loc="lower left")
        plt.tight_layout()
        plt.show()

        print("PR-AUC:", pr_auc)
        print("ROC-AUC:", roc_auc)
        print("y_val samples:", len(y_val))
        print("y_proba samples:", len(y_proba))
    _()
    return


@app.cell
def _(val_labels_path, val_predictions_path):
    from scepsis_prediction import evaluation

    utility = evaluation.evaluate_sepsis_score(
        label_csv=val_labels_path,
        prediction_csv=val_predictions_path,
    )

    print(f"Validation utility score: {utility:.6f}")
    return


@app.cell
def _(df_test, patient_col, target_col, test_preds, val_df, y_val_pred):
    # Label the unlabeled test set with model predictions
    df_test_labeled = df_test.copy()
    df_test_labeled["SepsisLabel"] = test_preds.astype(int)

    testset_labeled_path = "data/testset_labeled_predictions.csv"
    predictions_path = "predictions.csv"

    df_test_labeled.to_csv(testset_labeled_path, index=False)
    df_test_labeled[["Patient_ID", "SepsisLabel"]].to_csv(predictions_path, index=False)

    # Build validation label/prediction CSVs so the utility can be computed
    val_labels_path = "data/validation_labels.csv"
    val_predictions_path = "data/validation_predictions.csv"

    validation_df = val_df[[patient_col, target_col]].copy()
    validation_df["PredictedSepsisLabel"] = y_val_pred.astype(int)

    validation_df[[patient_col, target_col]].rename(
        columns={patient_col: "Patient_ID", target_col: "SepsisLabel"}
    ).to_csv(val_labels_path, index=False)

    validation_df[[patient_col, "PredictedSepsisLabel"]].rename(
        columns={patient_col: "Patient_ID", "PredictedSepsisLabel": "SepsisLabel"}
    ).to_csv(val_predictions_path, index=False)

    print(f"Saved labeled test set to: {testset_labeled_path}")
    print(f"Saved test predictions to: {predictions_path}")
    print(f"Saved validation labels to: {val_labels_path}")
    print(f"Saved validation predictions to: {val_predictions_path}")
    return val_labels_path, val_predictions_path


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
