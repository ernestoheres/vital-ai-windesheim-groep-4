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
    features = [
        "Calcium",
        "Hgb",
        "EtCO2",
        "BUN"
    ]
    return (features,)


@app.cell
def _(df, df_test, features):
    from xgboost import XGBClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

    target_col = "SepsisLabel"

    # Clean column names just in case
    df.columns = df.columns.str.strip()
    df_test.columns = df_test.columns.str.strip()

    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()

    # split labeled data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # fill missing values using only train stats
    medians = X_train[features].median()
    X_train[features] = X_train[features].fillna(medians)
    X_val[features] = X_val[features].fillna(medians)

    # hospital test set: no labels
    X_test = df_test.copy()
    X_test[features] = X_test[features].fillna(medians)

    model = XGBClassifier(
        random_state=42,
        scale_pos_weight=39,  # ~(1 - 0.025) / 0.025
        device="cuda",
         max_depth=6,                  # 4-8, deeper = more complex patterns
        min_child_weight=10,          # higher = more conservative, good for imbalance

        # randomness / overfitting
        subsample=0.8,                # row sampling per tree
        colsample_bytree=0.8,         # feature sampling per tree

        # learning
        n_estimators=500,             # trees, use early stopping instead of guessing
        learning_rate=0.05,           # lower = better generalisation, needs more trees

        # regularisation
        reg_alpha=0.1,                # L1 — drives weak features to zero
        reg_lambda=1.0,               # L2 — default, usually fine
        gamma=0.1,                    # min loss reduction to split — pruning

        # performance
        tree_method='hist',           # fast on large datasets, use 'gpu_hist' if you have GPU
        eval_metric='aucpr',          # PR-AUC as your eval — matches your actual goal
        early_stopping_rounds=50,     # stops when val score stops improving
        n_jobs=-1
    )

    model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
        verbose=50)

    # local validation
    y_val_pred = model.predict(X_val)

    print("Accuracy:", accuracy_score(y_val, y_val_pred))
    print(confusion_matrix(y_val, y_val_pred))
    print(classification_report(y_val, y_val_pred))

    # final hospital predictions
    test_preds = model.predict(X_test)
    return X_val, model, test_preds, y_val, y_val_pred


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
def _(X_val, df_test, pd, test_preds, y_val, y_val_pred):
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

    validation_df = pd.DataFrame({
        "Patient_ID": X_val["Patient_ID"].astype(int).to_numpy(),
        "SepsisLabel": y_val.astype(int).to_numpy(),
        "PredictedSepsisLabel": y_val_pred.astype(int),
    })

    validation_df[["Patient_ID", "SepsisLabel"]].to_csv(val_labels_path, index=False)
    validation_df[["Patient_ID", "PredictedSepsisLabel"]].rename(
        columns={"PredictedSepsisLabel": "SepsisLabel"}
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
