import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Scepsis voorspellen tot 6 uur voor diagnose

    We bouwen eerst een uitlegbare baseline en testen daarna in een tweede CRISP-DM-cyclus of temporale signalen de utility verhogen.
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


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
    3. CRISP-DM cyclus 1: baseline model
    4. CRISP-DM cyclus 2: temporale features + ablatie
    5. Threshold optimaliseren op utility en predictions exporteren
    """)
    return


@app.cell
def _():

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from sklearn.metrics import average_precision_score
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split as tts_model
    from xgboost import XGBClassifier

    from scepsis_prediction import evaluation as eval_utils


    return (
        XGBClassifier,
        average_precision_score,
        confusion_matrix,
        eval_utils,
        np,
        pd,
        plt,
        precision_recall_curve,
        roc_auc_score,
        sns,
        tts_model,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1) Data inladen

    We laden train- en testdata in en controleren snel de vorm, kolommen en basisstatistiek.
    """)
    return


@app.cell
def _(pd):
    df = pd.read_csv("data/train_data.csv")
    df_test = pd.read_csv("data/test_data.csv")
    return df, df_test


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    We bekijken de ruwe numerieke verdelingen als eerste sanity check.
    """)
    return


@app.cell
def _(df):
    df.describe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    We toetsen welke numerieke features lineair meebewegen met het label.
    """)
    return


@app.cell
def _(df, plt, sns):
    corr = df.corr(numeric_only=True)["SepsisLabel"].sort_values(ascending=False)
    corr = corr.drop("SepsisLabel")

    plt.figure(figsize=(6, 10))
    sns.heatmap(corr.to_frame(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation with SepsisLabel")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    We controleren missende waarden en class imbalance om de modelkeuzes te onderbouwen.
    """)
    return


@app.cell
def _(df):
    missingness = df.isna().mean().sort_values(ascending=False).rename("missing_ratio")
    class_balance = df["SepsisLabel"].value_counts(normalize=True).rename("ratio")

    print("Top 10 missende features:")
    print(missingness.head(10))
    print("\nClass balance (SepsisLabel):")
    print(class_balance)

    missingness.head(15)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # CRISP-DM Cyclus 1

    In de eerste cyclus bouwen we een stabiele baseline met patient-level split en ruwe metingen.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Business understanding: de baseline moet vroeg signaleren zonder patient leakage.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Data understanding: we kiezen alleen features die direct beschikbaar zijn in de meetreeks.
    """)
    return


@app.cell
def _(df):
    drop_cols = {"SepsisLabel", "Patient_ID", "Unnamed: 0"}
    base_features = [col for col in df.columns if col not in drop_cols]
    return (base_features,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    We zetten vaste kolomnamen centraal zodat dezelfde helpers in beide cycli bruikbaar zijn.
    """)
    return


@app.cell
def _():
    patient_col = "Patient_ID"
    target_col = "SepsisLabel"
    time_col = "Hour"
    return patient_col, target_col, time_col


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    We splitsen op patientniveau en houden de sepsisverdeling gelijk tussen train en validatie.
    """)
    return


@app.cell
def _(tts_model):
    def split_patient_frames(df, patient_col, target_col):
        patient_level = df.groupby(patient_col)[target_col].max().reset_index()
        train_ids, val_ids = tts_model(
            patient_level[patient_col],
            test_size=0.2,
            random_state=42,
            stratify=patient_level[target_col],
        )
        train_df = df[df[patient_col].isin(train_ids)].copy()
        val_df = df[df[patient_col].isin(val_ids)].copy()
        return train_df, val_df

    return (split_patient_frames,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    We maken een herbruikbare modelrunner zodat baseline en ablatie exact dezelfde trainingslogica delen.
    """)
    return


@app.cell
def _(XGBClassifier, average_precision_score, np, roc_auc_score):
    def build_xgb_model(scale_pos_weight):
        return XGBClassifier(
            random_state=42,
            n_estimators=100,
            learning_rate=0.05,
            max_depth=4,
            scale_pos_weight=scale_pos_weight,
            subsample=0.8,
            tree_method="hist",
            device="cuda",
            eval_metric="logloss",
            n_jobs=-1,
        )


    def summarize_scores(y_true, y_proba):
        return {
            "pr_auc": float(average_precision_score(y_true, y_proba)),
            "roc_auc": float(roc_auc_score(y_true, y_proba)),
            "positive_rate": float(np.mean(y_true)),
        }


    return build_xgb_model, summarize_scores


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    We voegen een generieke feature-builder toe; baseline gebruikt de identiteitsversie.
    """)
    return


@app.function
def identity_builder(frame, patient_col, time_col):
    return frame.copy(), []


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Deze helper traint een model, maakt validatiekansen en bewaart alles wat we later voor utility en export nodig hebben.
    """)
    return


@app.cell
def _(build_xgb_model, summarize_scores):
    def run_experiment(
        train_df,
        val_df,
        test_df,
        base_features,
        patient_col,
        target_col,
        time_col,
        feature_builder,
        experiment_name,
    ):
        train_ready, extra_features = feature_builder(train_df, patient_col, time_col)
        val_ready, _ = feature_builder(val_df, patient_col, time_col)
        test_ready, _ = feature_builder(test_df, patient_col, time_col)

        train_ready = train_ready.replace([float("inf"), float("-inf")], float("nan"))
        val_ready = val_ready.replace([float("inf"), float("-inf")], float("nan"))
        test_ready = test_ready.replace([float("inf"), float("-inf")], float("nan"))

        model_features = base_features + extra_features
        y_train = train_ready[target_col].copy()
        y_val = val_ready[target_col].copy()

        model = build_xgb_model(scale_pos_weight=8.75)
        model.fit(train_ready[model_features], y_train, eval_set=[(val_ready[model_features], y_val)], verbose=False)
        y_val_proba = model.predict_proba(val_ready[model_features])[:, 1]

        return {
            "name": experiment_name,
            "features": model_features,
            "model": model,
            "scores": summarize_scores(y_val, y_val_proba),
            "test_ready": test_ready,
            "val_df": val_ready,
            "X_test": test_ready[model_features].copy(),
            "y_val": y_val,
            "y_val_proba": y_val_proba,
        }

    return (run_experiment,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Modeling: we maken eerst een enkele patient-level split die in beide cycli hergebruikt wordt.
    """)
    return


@app.cell
def _(df, df_test, patient_col, split_patient_frames, target_col):
    df_model = df.copy()
    df_test_model = df_test.copy()
    df_model.columns = df_model.columns.str.strip()
    df_test_model.columns = df_test_model.columns.str.strip()

    train_df, val_df = split_patient_frames(df_model, patient_col, target_col)
    return df_test_model, train_df, val_df


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    We draaien de baseline als referentiepunt voor alle latere temporale experimenten.
    """)
    return


@app.cell
def _(
    base_features,
    df_test_model,
    patient_col,
    run_experiment,
    target_col,
    time_col,
    train_df,
    val_df,
):
    baseline_run = run_experiment(
        train_df=train_df,
        val_df=val_df,
        test_df=df_test_model,
        base_features=base_features,
        patient_col=patient_col,
        target_col=target_col,
        time_col=time_col,
        feature_builder=identity_builder,
        experiment_name="cycle_1_baseline",
    )
    baseline_run
    return (baseline_run,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    We controleren direct de baseline-kwaliteit voordat we extra complexiteit toevoegen.
    """)
    return


@app.cell
def _(baseline_run, pd):
    print(f"Aantal baseline features: {len(baseline_run['features'])}")
    print(f"Baseline utility-threshold startpunt: 0.5")
    pd.DataFrame([baseline_run["scores"]], index=[baseline_run["name"]])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # CRISP-DM Cyclus 2

    In de tweede cyclus testen we de hypothese dat patient-progressie extra utility oplevert.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Business understanding: we focussen op vroege verslechtering via bredere trendfeatures op vitals en enkele kernlabs.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Data understanding: temporale features moeten patient-safe zijn, alleen verleden gebruiken en geen ruisexplosie veroorzaken.
    """)
    return


@app.cell
def _(df):
    fill_forward_raw = [
        "HR",
        "O2Sat",
        "Temp",
        "SBP",
        "MAP",
        "DBP",
        "Resp",
        "Lactate",
        "Creatinine",
        "WBC",
        "Platelets",
        "Bilirubin_total",
        "BUN",
    ]
    vital_roll_raw = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp"]
    oldstyle_roll_raw = [
        "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2",
        "Glucose", "Lactate", "Creatinine", "WBC", "Platelets", "BUN",
        "FiO2", "SaO2", "PaCO2", "pH",
    ]
    count_feature_raw = [
        "Temp", "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST",
        "BUN", "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
        "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium", "Bilirubin_total",
        "TroponinI", "Hct", "Hgb", "PTT", "WBC", "Fibrinogen", "Platelets",
    ]
    partial_sofa_raw = ["Platelets", "Bilirubin_total", "MAP", "Creatinine"]

    fill_forward_cols = [col for col in fill_forward_raw if col in df.columns]
    vital_roll_cols = [col for col in vital_roll_raw if col in df.columns]
    count_feature_cols = [col for col in count_feature_raw if col in df.columns]
    partial_sofa_cols = [col for col in partial_sofa_raw if col in df.columns]
    return (
        count_feature_cols,
        fill_forward_cols,
        oldstyle_roll_raw,
        partial_sofa_cols,
        vital_roll_cols,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    We bouwen compacte temporal helpers: eerst patient-wise sorteren en forward fillen, daarna bredere trendfeatures maken.
    """)
    return


@app.cell
def _():
    def sort_patient_time(frame, patient_col, time_col):
        return frame.sort_values([patient_col, time_col]).copy()

    def forward_fill_by_patient(frame, patient_col, cols):
        if not cols:
            return frame
        filled = frame.groupby(patient_col)[cols].ffill()
        frame.loc[:, cols] = filled
        return frame

    return forward_fill_by_patient, sort_patient_time


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    We houden de rolling helpers klein zodat de temporale logica leesbaar blijft per experiment.
    """)
    return


@app.cell
def _():
    def make_roll_mean_features(frame, patient_col, cols, windows):
        feature_map = {}
        valid_cols = [col for col in cols if col in frame.columns]
        for col in valid_cols:
            grouped = frame.groupby(patient_col)[col]
            for window in windows:
                feature_name = f"{col}_roll{window}_mean"
                feature_map[feature_name] = grouped.transform(
                    lambda s: s.rolling(window, min_periods=1).mean()
                )
        return feature_map

    def make_roll_std_features(frame, patient_col, cols, windows):
        feature_map = {}
        valid_cols = [col for col in cols if col in frame.columns]
        for col in valid_cols:
            grouped = frame.groupby(patient_col)[col]
            for window in windows:
                feature_name = f"{col}_roll{window}_std"
                feature_map[feature_name] = grouped.transform(
                    lambda s: s.rolling(window, min_periods=2).std()
                )
        return feature_map

    def make_roll_min_max_features(frame, patient_col, cols, windows):
        feature_map = {}
        valid_cols = [col for col in cols if col in frame.columns]
        for col in valid_cols:
            grouped = frame.groupby(patient_col)[col]
            for window in windows:
                min_name = f"{col}_roll{window}_min"
                max_name = f"{col}_roll{window}_max"
                feature_map[min_name] = grouped.transform(
                    lambda s: s.rolling(window, min_periods=1).min()
                )
                feature_map[max_name] = grouped.transform(
                    lambda s: s.rolling(window, min_periods=1).max()
                )
        return feature_map

    def make_measurement_count_features(frame, patient_col, cols, window):
        feature_map = {}
        valid_cols = [col for col in cols if col in frame.columns]
        for col in valid_cols:
            feature_name = f"{col}_count_{window}"
            feature_map[feature_name] = frame.groupby(patient_col)[col].transform(
                lambda s: s.notna().astype(int).rolling(window, min_periods=1).sum()
            )
        return feature_map

    return (
        make_measurement_count_features,
        make_roll_mean_features,
        make_roll_min_max_features,
        make_roll_std_features,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    We maken meerdere temporale builders zodat fill-forward en extra featurefamilies eerlijk tegen elkaar getest worden.
    """)
    return


@app.cell
def _(
    count_feature_cols,
    fill_forward_cols,
    forward_fill_by_patient,
    make_measurement_count_features,
    make_roll_mean_features,
    make_roll_min_max_features,
    make_roll_std_features,
    oldstyle_roll_raw,
    partial_sofa_cols,
    pd,
    sort_patient_time,
    vital_roll_cols,
):
    def add_extra_engineered_features(ready):
        if {"HR", "SBP"}.issubset(ready.columns):
            ready["shock_index"] = ready["HR"].div(ready["SBP"].replace(0, float("nan")))

        if {"BUN", "Creatinine"}.issubset(ready.columns):
            ready["bun_creatinine_ratio"] = ready["BUN"].div(ready["Creatinine"].replace(0, float("nan")))

        if len(partial_sofa_cols) == 4:
            platelet_score = pd.Series(0.0, index=ready.index)
            bilirubin_score = pd.Series(0.0, index=ready.index)
            map_score = pd.Series(0.0, index=ready.index)
            creatinine_score = pd.Series(0.0, index=ready.index)

            platelet_score = platelet_score.mask(ready["Platelets"] < 150, 1.0)
            platelet_score = platelet_score.mask(ready["Platelets"] < 100, 2.0)
            platelet_score = platelet_score.mask(ready["Platelets"] < 50, 3.0)
            platelet_score = platelet_score.mask(ready["Platelets"] < 20, 4.0)

            bilirubin_score = bilirubin_score.mask(ready["Bilirubin_total"] >= 1.2, 1.0)
            bilirubin_score = bilirubin_score.mask(ready["Bilirubin_total"] >= 2.0, 2.0)
            bilirubin_score = bilirubin_score.mask(ready["Bilirubin_total"] >= 6.0, 3.0)
            bilirubin_score = bilirubin_score.mask(ready["Bilirubin_total"] >= 12.0, 4.0)

            map_score = map_score.mask(ready["MAP"] < 70, 1.0)

            creatinine_score = creatinine_score.mask(ready["Creatinine"] >= 1.2, 1.0)
            creatinine_score = creatinine_score.mask(ready["Creatinine"] >= 2.0, 2.0)
            creatinine_score = creatinine_score.mask(ready["Creatinine"] >= 3.5, 3.0)
            creatinine_score = creatinine_score.mask(ready["Creatinine"] >= 5.0, 4.0)

            ready["partial_sofa"] = platelet_score + bilirubin_score + map_score + creatinine_score

        return ready

    def build_broad_temporal(frame, patient_col, time_col, use_ffill):
        ready = sort_patient_time(frame, patient_col, time_col)
        if use_ffill:
            fill_cols = [col for col in fill_forward_cols if col in ready.columns]
            ready = forward_fill_by_patient(ready, patient_col, fill_cols)

        ready = add_extra_engineered_features(ready)

        feature_map = {}
        feature_map.update(make_roll_mean_features(ready, patient_col, vital_roll_cols, [7]))
        feature_map.update(make_roll_std_features(ready, patient_col, vital_roll_cols, [7]))
        feature_map.update(make_roll_min_max_features(ready, patient_col, vital_roll_cols, [6]))
        feature_map.update(make_measurement_count_features(ready, patient_col, count_feature_cols, 8))

        core_temporal_cols = [
            col for col in ["shock_index", "bun_creatinine_ratio", "partial_sofa"]
            if col in ready.columns
        ]
        feature_map.update(make_roll_mean_features(ready, patient_col, core_temporal_cols, [7]))
        feature_map.update(make_roll_std_features(ready, patient_col, core_temporal_cols, [7]))

        temporal_df = pd.DataFrame(feature_map, index=ready.index)
        extra_features = core_temporal_cols + temporal_df.columns.tolist()
        ready = pd.concat([ready, temporal_df], axis=1)
        return ready, extra_features

    def build_oldstyle_temporal(frame, patient_col, time_col):
        ready = sort_patient_time(frame, patient_col, time_col)
        oldstyle_roll_cols = [col for col in oldstyle_roll_raw if col in ready.columns]

        feature_map = {}
        feature_map.update(make_roll_mean_features(ready, patient_col, oldstyle_roll_cols, [3, 6]))
        feature_map.update(make_roll_std_features(ready, patient_col, oldstyle_roll_cols, [3, 6]))
        feature_map.update(make_roll_min_max_features(ready, patient_col, oldstyle_roll_cols, [3, 6]))

        temporal_df = pd.DataFrame(feature_map, index=ready.index)
        extra_features = temporal_df.columns.tolist()
        ready = pd.concat([ready, temporal_df], axis=1)
        return ready, extra_features

    def temporal_broad_no_ffill_builder(frame, patient_col, time_col):
        return build_broad_temporal(frame, patient_col, time_col, use_ffill=False)

    def temporal_broad_with_ffill_builder(frame, patient_col, time_col):
        return build_broad_temporal(frame, patient_col, time_col, use_ffill=True)

    def temporal_oldstyle_broad_builder(frame, patient_col, time_col):
        return build_oldstyle_temporal(frame, patient_col, time_col)

    return (
        temporal_broad_no_ffill_builder,
        temporal_broad_with_ffill_builder,
        temporal_oldstyle_broad_builder,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Modeling: we vergelijken meerdere temporale varianten op exact dezelfde patient split.
    """)
    return


@app.cell
def _(
    base_features,
    df_test_model,
    patient_col,
    run_experiment,
    target_col,
    temporal_broad_no_ffill_builder,
    temporal_broad_with_ffill_builder,
    temporal_oldstyle_broad_builder,
    time_col,
    train_df,
    val_df,
):
    experiments = [
        ("temporal_broad_no_ffill", temporal_broad_no_ffill_builder),
        ("temporal_broad_with_ffill", temporal_broad_with_ffill_builder),
        ("temporal_oldstyle_broad", temporal_oldstyle_broad_builder),
    ]

    cycle_2_runs = []
    for name, builder in experiments:
        experiment_run_2 = run_experiment(
            train_df, val_df, df_test_model, base_features,
            patient_col, target_col, time_col, builder, name,
        )
        cycle_2_runs.append(experiment_run_2)
    return (cycle_2_runs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # CRISP-DM Cyclus 3

    In de derde cyclus testen we apart of meetpatronen en tijd-sinds-laatste-meting extra utility geven.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Business understanding: we toetsen nu niet de meetwaarde zelf, maar het observatiegedrag rond die meetwaarde.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Data understanding: we laten statische metadata buiten deze featurefamilie en herhalen forward fill alleen als losse ablatie.
    """)
    return


@app.cell
def _(df):
    missingness_raw = [
        "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2",
        "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
        "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
        "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
        "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC",
        "Fibrinogen", "Platelets",
    ]
    missingness_cols = [col for col in missingness_raw if col in df.columns]
    missingness_cols
    return (missingness_cols,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    We bouwen kleine helpers voor presence-features en tijd sinds laatste meting per patient.
    """)
    return


@app.cell
def _():
    def make_was_measured_features(frame, cols):
        feature_map = {}
        valid_cols = [col for col in cols if col in frame.columns]
        for col in valid_cols:
            feature_map[f"{col}_was_measured"] = frame[col].notna().astype(int)
        return feature_map


    def make_tslm_features(frame, patient_col, time_col, cols):
        feature_map = {}
        valid_cols = [col for col in cols if col in frame.columns]
        for col in valid_cols:
            last_seen = frame[time_col].where(frame[col].notna()).groupby(frame[patient_col]).ffill()
            feature_map[f"{col}_tslm"] = frame[time_col] - last_seen
        return feature_map


    return make_tslm_features, make_was_measured_features


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    We testen vier compacte varianten: indicatoren, tslm, gecombineerd en gecombineerd met forward fill.
    """)
    return


@app.cell
def _(
    fill_forward_cols,
    forward_fill_by_patient,
    make_tslm_features,
    make_was_measured_features,
    missingness_cols,
    pd,
    sort_patient_time,
):
    def build_missingness_feature_frame(frame, patient_col, time_col, include_presence, include_tslm):
        feature_map = {}
        if include_presence:
            feature_map.update(make_was_measured_features(frame, missingness_cols))
        if include_tslm:
            feature_map.update(make_tslm_features(frame, patient_col, time_col, missingness_cols))

        temporal_df = pd.DataFrame(feature_map, index=frame.index)
        extra_features = temporal_df.columns.tolist()
        return temporal_df, extra_features


    def build_temporal_missingness(frame, patient_col, time_col, use_ffill, include_presence, include_tslm):
        ready = sort_patient_time(frame, patient_col, time_col)
        if use_ffill:
            fill_cols = [col for col in fill_forward_cols if col in ready.columns]
            ready = forward_fill_by_patient(ready, patient_col, fill_cols)

        temporal_df, extra_features = build_missingness_feature_frame(
            ready,
            patient_col,
            time_col,
            include_presence=include_presence,
            include_tslm=include_tslm,
        )
        ready = pd.concat([ready, temporal_df], axis=1)
        return ready, extra_features


    def missingness_only_builder(frame, patient_col, time_col):
        return build_temporal_missingness(
            frame, patient_col, time_col, use_ffill=False, include_presence=True, include_tslm=False
        )


    def tslm_only_builder(frame, patient_col, time_col):
        return build_temporal_missingness(
            frame, patient_col, time_col, use_ffill=False, include_presence=False, include_tslm=True
        )


    def missingness_tslm_builder(frame, patient_col, time_col):
        return build_temporal_missingness(
            frame, patient_col, time_col, use_ffill=False, include_presence=True, include_tslm=True
        )


    def missingness_tslm_ffill_builder(frame, patient_col, time_col):
        return build_temporal_missingness(
            frame, patient_col, time_col, use_ffill=True, include_presence=True, include_tslm=True
        )


    return (
        build_missingness_feature_frame,
        missingness_only_builder,
        missingness_tslm_builder,
        missingness_tslm_ffill_builder,
        tslm_only_builder,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Modeling: cycle 3 gebruikt dezelfde patient split en dezelfde utility-workflow als de eerdere cycli.
    """)
    return


@app.cell
def _(
    base_features,
    df_test_model,
    missingness_only_builder,
    missingness_tslm_builder,
    missingness_tslm_ffill_builder,
    patient_col,
    run_experiment,
    target_col,
    time_col,
    train_df,
    tslm_only_builder,
    val_df,
):
    cycle_3_experiments = [
        ("cycle_3_missingness_only", missingness_only_builder),
        ("cycle_3_tslm_only", tslm_only_builder),
        ("cycle_3_missingness_tslm", missingness_tslm_builder),
        ("cycle_3_missingness_tslm_ffill", missingness_tslm_ffill_builder),
    ]

    cycle_3_runs = []
    for cycle_3_name, cycle_3_builder in cycle_3_experiments:
        cycle_3_run = run_experiment(
            train_df, val_df, df_test_model, base_features,
            patient_col, target_col, time_col, cycle_3_builder, cycle_3_name,
        )
        cycle_3_runs.append(cycle_3_run)

    cycle_3_runs
    return (cycle_3_runs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    We stapelen daarna missingness-signalen op de beste brede cycle 2-builder om te toetsen of ze nog extra utility toevoegen.
    """)
    return


@app.cell
def _(build_missingness_feature_frame, pd, temporal_broad_with_ffill_builder):
    def build_cycle_2_plus_missingness(
        frame,
        patient_col,
        time_col,
        include_presence,
        include_tslm,
    ):
        ready, cycle_2_features = temporal_broad_with_ffill_builder(frame, patient_col, time_col)
        missing_df, missing_features = build_missingness_feature_frame(
            ready,
            patient_col,
            time_col,
            include_presence=include_presence,
            include_tslm=include_tslm,
        )
        ready = pd.concat([ready, missing_df], axis=1)
        return ready, cycle_2_features + missing_features


    def cycle_2_best_plus_missingness_builder(frame, patient_col, time_col):
        return build_cycle_2_plus_missingness(
            frame,
            patient_col,
            time_col,
            include_presence=True,
            include_tslm=False,
        )


    def cycle_2_best_plus_tslm_builder(frame, patient_col, time_col):
        return build_cycle_2_plus_missingness(
            frame,
            patient_col,
            time_col,
            include_presence=False,
            include_tslm=True,
        )


    def cycle_2_best_plus_missingness_tslm_ffill_builder(frame, patient_col, time_col):
        return build_cycle_2_plus_missingness(
            frame,
            patient_col,
            time_col,
            include_presence=True,
            include_tslm=True,
        )


    return (
        cycle_2_best_plus_missingness_builder,
        cycle_2_best_plus_missingness_tslm_ffill_builder,
        cycle_2_best_plus_tslm_builder,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    We testen drie gestapelde ablaties: cycle 2 plus presence, plus tslm en plus beide samen.
    """)
    return


@app.cell
def _(
    base_features,
    cycle_2_best_plus_missingness_builder,
    cycle_2_best_plus_missingness_tslm_ffill_builder,
    cycle_2_best_plus_tslm_builder,
    df_test_model,
    patient_col,
    run_experiment,
    target_col,
    time_col,
    train_df,
    val_df,
):
    combined_experiments = [
        ("cycle_2_best_plus_missingness", cycle_2_best_plus_missingness_builder),
        ("cycle_2_best_plus_tslm", cycle_2_best_plus_tslm_builder),
        ("cycle_2_best_plus_missingness_tslm_ffill", cycle_2_best_plus_missingness_tslm_ffill_builder),
    ]

    combined_cycle_runs = []
    for combined_name, combined_builder in combined_experiments:
        combined_run = run_experiment(
            train_df, val_df, df_test_model, base_features,
            patient_col, target_col, time_col, combined_builder, combined_name,
        )
        combined_cycle_runs.append(combined_run)

    combined_cycle_runs
    return (combined_cycle_runs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    We tonen ook los de cycle 3-varianten, zodat missingness-signalen apart leesbaar blijven.
    """)
    return


@app.cell
def _(ablation_results):
    cycle_3_results = ablation_results[
        ablation_results["experiment"].str.startswith("cycle_3_")
    ].reset_index(drop=True)
    cycle_3_results
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    We tonen ook apart de gestapelde runs, zodat zichtbaar blijft of missingness bovenop cycle 2 nog helpt.
    """)
    return


@app.cell
def _(ablation_results):
    combined_results = ablation_results[
        ablation_results["experiment"].str.startswith("cycle_2_best_plus_")
    ].reset_index(drop=True)
    combined_results
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    We vergelijken baseline en temporale uitbreiding eerst snel op vaste metrics en een standaard threshold van 0.5.
    """)
    return


@app.cell
def _():
    default_threshold = 0.5
    return (default_threshold,)


@app.cell
def _(
    baseline_run,
    combined_cycle_runs,
    cycle_2_runs,
    cycle_3_runs,
    default_threshold,
    evaluate_utility_at_threshold,
    patient_col,
    pd,
    target_col,
):
    all_runs = [baseline_run] + cycle_2_runs + cycle_3_runs + combined_cycle_runs
    rows = []
    for run_item in all_runs:
        default_pred = (run_item["y_val_proba"] >= default_threshold).astype(int)
        utility_at_default = evaluate_utility_at_threshold(
            run_item, patient_col, target_col, default_threshold
        )
        run_item["default_threshold"] = default_threshold
        run_item["default_utility"] = utility_at_default
        rows.append({
            "experiment": run_item["name"],
            **run_item["scores"],
            "n_features": len(run_item["features"]),
            "threshold": default_threshold,
            "pred_pos_rate": float(default_pred.mean()),
            "utility_at_0_5": utility_at_default,
        })

    ablation_results = pd.DataFrame(rows)
    ablation_results = ablation_results.sort_values(["utility_at_0_5", "pr_auc"], ascending=False)
    ablation_results = ablation_results.reset_index(drop=True)
    ablation_results
    return ablation_results, all_runs


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    We berekenen eerst utility op threshold 0.5 en sturen daarna de top 3 runs door naar een grove en fijne sweep.
    """)
    return


@app.cell
def _(eval_utils, np):
    def evaluate_utility_at_threshold(run, patient_col, target_col, threshold):
        val_eval_df = run["val_df"][[patient_col, target_col]].copy().reset_index(drop=True)
        val_eval_df["proba"] = run["y_val_proba"]

        label_path = f"data/{run['name']}_labels.csv"
        pred_path = f"data/{run['name']}_predictions.csv"
        val_eval_df[[patient_col, target_col]].to_csv(label_path, index=False)

        preds = val_eval_df[[patient_col]].copy()
        preds["SepsisLabel"] = (val_eval_df["proba"] >= threshold).astype(int)
        preds.to_csv(pred_path, index=False)
        score = eval_utils.evaluate_sepsis_score(label_csv=label_path, prediction_csv=pred_path)
        return float(score)

    def select_best_threshold(run, patient_col, target_col):
        val_eval_df = run["val_df"][[patient_col, target_col]].copy().reset_index(drop=True)
        val_eval_df["proba"] = run["y_val_proba"]
        thresholds = np.arange(0.10, 1.00, 0.05)

        label_path = f"data/{run['name']}_labels.csv"
        pred_path = f"data/{run['name']}_predictions.csv"
        val_eval_df[[patient_col, target_col]].to_csv(label_path, index=False)

        utilities = []
        for threshold in thresholds:
            preds = val_eval_df[[patient_col]].copy()
            preds["SepsisLabel"] = (val_eval_df["proba"] >= threshold).astype(int)
            preds.to_csv(pred_path, index=False)
            score = eval_utils.evaluate_sepsis_score(label_csv=label_path, prediction_csv=pred_path)
            utilities.append(score)

        best_idx = int(np.argmax(utilities))
        return float(thresholds[best_idx]), float(utilities[best_idx]), thresholds, utilities

    return evaluate_utility_at_threshold, select_best_threshold


@app.function
def select_fine_threshold(eval_utils, np, run, patient_col, target_col):
    val_eval_df = run["val_df"][[patient_col, target_col]].copy().reset_index(drop=True)
    val_eval_df["proba"] = run["y_val_proba"]
    thresholds = np.arange(0.38, 0.501, 0.005)

    label_path = f"data/{run['name']}_labels.csv"
    pred_path = f"data/{run['name']}_predictions.csv"
    val_eval_df[[patient_col, target_col]].to_csv(label_path, index=False)

    utilities = []
    for threshold in thresholds:
        preds = val_eval_df[[patient_col]].copy()
        preds["SepsisLabel"] = (val_eval_df["proba"] >= threshold).astype(int)
        preds.to_csv(pred_path, index=False)
        score = eval_utils.evaluate_sepsis_score(label_csv=label_path, prediction_csv=pred_path)
        utilities.append(score)

    best_idx = int(np.argmax(utilities))
    return float(thresholds[best_idx]), float(utilities[best_idx]), thresholds, utilities


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    We nemen de top 3 runs op utility bij threshold 0.5 en tonen de grove en fijne sweep apart.
    """)
    return


@app.cell
def _(
    ablation_results,
    all_runs,
    default_threshold,
    eval_utils,
    np,
    patient_col,
    pd,
    select_best_threshold,
    target_col,
):
    top_experiment_names = ablation_results.head(3)["experiment"].tolist()
    shortlisted_runs = [run for run in all_runs if run["name"] in top_experiment_names]

    coarse_rows = []
    for shortlisted_run in shortlisted_runs:
        best_threshold, best_utility, sweep_thresholds, sweep_utilities = select_best_threshold(
            shortlisted_run, patient_col, target_col
        )
        shortlisted_run["best_threshold"] = best_threshold
        shortlisted_run["best_utility"] = best_utility
        shortlisted_run["threshold_grid"] = sweep_thresholds
        shortlisted_run["utility_curve"] = sweep_utilities
        coarse_rows.append({
            "experiment": shortlisted_run["name"],
            "default_threshold": default_threshold,
            "utility_at_0_5": shortlisted_run["default_utility"],
            "coarse_best_threshold": best_threshold,
            "coarse_best_utility": best_utility,
        })

    coarse_threshold_results = pd.DataFrame(coarse_rows)
    coarse_threshold_results = coarse_threshold_results.sort_values(
        ["coarse_best_utility", "utility_at_0_5"], ascending=False
    )
    coarse_threshold_results = coarse_threshold_results.reset_index(drop=True)
    coarse_threshold_results

    fine_rows = []
    for shortlisted_run in shortlisted_runs:
        fine_threshold, fine_utility, fine_thresholds, fine_utilities = select_fine_threshold(
            eval_utils, np, shortlisted_run, patient_col, target_col
        )
        shortlisted_run["fine_best_threshold"] = fine_threshold
        shortlisted_run["fine_best_utility"] = fine_utility
        shortlisted_run["fine_threshold_grid"] = fine_thresholds
        shortlisted_run["fine_utility_curve"] = fine_utilities
        shortlisted_run["final_threshold"] = (
            fine_threshold if fine_utility >= shortlisted_run["best_utility"] else shortlisted_run["best_threshold"]
        )
        shortlisted_run["final_utility"] = max(fine_utility, shortlisted_run["best_utility"])
        fine_rows.append({
            "experiment": shortlisted_run["name"],
            "coarse_best_threshold": shortlisted_run["best_threshold"],
            "coarse_best_utility": shortlisted_run["best_utility"],
            "fine_best_threshold": fine_threshold,
            "fine_best_utility": fine_utility,
            "final_threshold": shortlisted_run["final_threshold"],
            "final_utility": shortlisted_run["final_utility"],
        })

    fine_threshold_results = pd.DataFrame(fine_rows)
    fine_threshold_results = fine_threshold_results.sort_values(
        ["final_utility", "fine_best_utility", "coarse_best_utility"], ascending=False
    )
    fine_threshold_results = fine_threshold_results.reset_index(drop=True)
    fine_threshold_results

    best_run_name = fine_threshold_results.loc[0, "experiment"]
    best_run = next(run for run in shortlisted_runs if run["name"] == best_run_name)
    return (
        best_run,
        coarse_threshold_results,
        fine_threshold_results,
        shortlisted_runs,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    We tonen de utilitytabellen expliciet zodat de beste run, de grove sweep en de fijne sweep direct afleesbaar zijn.
    """)
    return


@app.cell
def _(
    best_run,
    coarse_threshold_results,
    fine_threshold_results,
    shortlisted_runs,
):
    print("Top 3 runs voor threshold-optimalisatie:")
    print(", ".join(run["name"] for run in shortlisted_runs))
    print("Beste run na sweep:", best_run["name"])
    print("Utility op 0.5:", round(float(best_run["default_utility"]), 6))
    print("Beste utility na grove sweep:", round(float(best_run["best_utility"]), 6))
    print("Beste threshold na grove sweep:", round(float(best_run["best_threshold"]), 3))
    print("Beste utility na fijne sweep:", round(float(best_run["fine_best_utility"]), 6))
    print("Beste threshold na fijne sweep:", round(float(best_run["fine_best_threshold"]), 3))
    print("Gebruikte eindthreshold:", round(float(best_run["final_threshold"]), 3))
    coarse_threshold_results
    fine_threshold_results
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    We visualiseren ook de fijne sweep tussen 0.38 en 0.50 om kleine utilityverschillen zichtbaar te maken.
    """)
    return


@app.cell
def _(best_run, plt):
    plt.figure(figsize=(10, 5))
    plt.plot(best_run["fine_threshold_grid"], best_run["fine_utility_curve"])
    plt.axvline(best_run["fine_best_threshold"], linestyle="--")
    plt.xlabel("Threshold")
    plt.ylabel("Validation utility")
    plt.title(f"Fine utility sweep - {best_run['name']}")
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    We visualiseren de utility-curve van de beste run om de thresholdkeuze controleerbaar te houden.
    """)
    return


@app.cell
def _(best_run, plt):
    plt.figure(figsize=(10, 5))
    plt.plot(best_run["threshold_grid"], best_run["utility_curve"])
    plt.axvline(best_run["best_threshold"], linestyle="--")
    plt.xlabel("Threshold")
    plt.ylabel("Validation utility")
    plt.title(f"Utility vs threshold - {best_run['name']}")
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    We inspecteren de confusion matrix van de utility-optimum in plaats van de standaard 0.5-threshold.
    """)
    return


@app.cell
def _(best_run, confusion_matrix, plt, sns):
    best_run_pred = (best_run["y_val_proba"] >= best_run["final_threshold"]).astype(int)
    cm = confusion_matrix(best_run["y_val"], best_run_pred)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {best_run['name']}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    We tonen de PR-curve van de beste run omdat PR vaak beter past bij deze ongebalanceerde taak.
    """)
    return


@app.cell
def _(best_run, plt, precision_recall_curve):
    precision, recall, _ = precision_recall_curve(best_run["y_val"], best_run["y_val_proba"])

    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision, label=f"PR-AUC = {best_run['scores']['pr_auc']:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve - {best_run['name']}")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    We exporteren tenslotte predictions met de beste utility-threshold uit de notebook.
    """)
    return


@app.cell
def _(best_run, patient_col):
    test_proba = best_run["model"].predict_proba(best_run["X_test"])[:, 1]
    test_preds = (test_proba >= best_run["final_threshold"]).astype(int)

    submission_df = best_run["test_ready"][[patient_col]].copy()
    submission_df["SepsisLabel"] = test_preds
    submission_df.to_csv("predictions.csv", index=False)

    print("Beste experiment:", best_run["name"])
    print("Best threshold:", round(float(best_run["final_threshold"]), 3))
    print("Predictions opgeslagen in predictions.csv")
    submission_df.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Reflectie

    - Cyclus 1 geeft een stabiele baseline zonder leakage.
    - Cyclus 2 toetst gericht of progressiesignalen extra utility leveren.
    - Cyclus 3 toetst apart of temporale missingness-signalen genoeg extra waarde geven.
    - De uiteindelijke keuze blijft utility-gedreven in plaats van threshold 0.5-gedreven.
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
