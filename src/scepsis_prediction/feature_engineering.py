"""
Feature Engineering voor Sepsis Predictie
==========================================
Losse functies die elk op een DataFrame werken en een verrijkt
DataFrame teruggeven. Aanroepen naar keuze, in elke volgorde.

Gebruik:
    import pandas as pd
    from feature_engineering import (
        add_sofa_features,
        add_hemodynamic_features,
        add_respiratory_features,
        add_acid_base_features,
        add_renal_features,
        add_liver_coag_features,
        add_hematology_features,
        add_news2_score,
        add_temporal_features,
        add_rolling_features,
    )

    df = pd.read_csv("train_data.csv")
    df = add_sofa_features(df)
    df = add_hemodynamic_features(df)
    # ... etc.
"""

from scepsis_prediction.sofa_calculator import SofaCalculator
import pandas as pd
import numpy as np

def __has(df: pd.DataFrame, *cols) -> bool:
    """Geeft True terug als alle opgegeven kolommen aanwezig zijn."""
    return set(cols).issubset(df.columns)

def add_sofa_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Berekent qSOFA en SOFA via SofaCalculator.
    Tussenresultaten (sofa_resp, SF_ratio, etc.) worden niet bewaard.

    Toegevoegde kolommen:
        qSOFA_partial        — 0–2 (zonder GCS)
        SOFA_modified_total  — som van resp, coag, lever, cv, renaal
    """

    return SofaCalculator(df).calculate_all_values(True, True)

def add_hemodynamic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Toegevoegde kolommen:
        ShockIndex           — HR / SBP
        ModifiedShockIndex   — HR / MAP
        PulsePressure        — SBP - DBP
        PP_MAP_ratio         — PulsePressure / MAP
        RatePressureProduct  — HR × SBP (myocardiaal O2-verbruik)
    """
    if __has(df, "HR", "SBP"):
        df["ShockIndex"] = df["HR"].div(df["SBP"].replace(0, np.nan))

    if __has(df, "HR", "MAP"):
        df["ModifiedShockIndex"] = df["HR"].div(df["MAP"].replace(0, np.nan))

    if __has(df, "SBP", "DBP"):
        df["PulsePressure"] = df["SBP"] - df["DBP"]
        if "MAP" in df.columns:
            df["PP_MAP_ratio"] = df["PulsePressure"].div(df["MAP"].replace(0, np.nan))

    if __has(df, "HR", "SBP"):
        df["RatePressureProduct"] = df["HR"] * df["SBP"]

    return df

def add_respiratory_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Toegevoegde kolommen:
        SF_ratio         — O2Sat / FiO2  (proxy voor PaO2/FiO2)
        SaO2_FiO2_ratio  — SaO2 / FiO2
        PaCO2_EtCO2_gap  — PaCO2 - EtCO2 (dood-ruimte ventilatie)
        VentilationIndex — Resp × PaCO2 / 100
    """
    if __has(df, "O2Sat", "FiO2"):
        df["SF_ratio"] = df["O2Sat"].div(df["FiO2"].replace(0, np.nan))

    if __has(df, "SaO2", "FiO2"):
        df["SaO2_FiO2_ratio"] = df["SaO2"].div(df["FiO2"].replace(0, np.nan))

    if __has(df, "PaCO2", "EtCO2"):
        df["PaCO2_EtCO2_gap"] = df["PaCO2"] - df["EtCO2"]

    if __has(df, "Resp", "PaCO2"):
        df["VentilationIndex"] = df["Resp"] * df["PaCO2"] / 100

    return df

def add_acid_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Toegevoegde kolommen:
        Cl_HCO3_sum          — Chloride + HCO3
        AnionGap_estimated   — 140 - (Cl + HCO3)  (Na ≈ 140 aangenomen)
        SevereAcidosis_flag  — BaseExcess < -8
        pH_HCO3_ratio        — pH / HCO3
        Lactate_HCO3_ratio   — Lactate / HCO3
    """
    if __has(df, "Chloride", "HCO3"):
        df["Cl_HCO3_sum"]        = df["Chloride"] + df["HCO3"]
        df["AnionGap_estimated"] = 140 - df["Cl_HCO3_sum"]

    if "BaseExcess" in df.columns:
        df["SevereAcidosis_flag"] = (df["BaseExcess"] < -8).astype(float)

    if __has(df, "pH", "HCO3"):
        df["pH_HCO3_ratio"] = df["pH"].div(df["HCO3"].replace(0, np.nan))

    if __has(df, "Lactate", "HCO3"):
        df["Lactate_HCO3_ratio"] = df["Lactate"].div(df["HCO3"].replace(0, np.nan))

    return df

def add_renal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Toegevoegde kolommen:
        BUN_Cr_ratio    — BUN / Creatinine  (prerenaal vs renaal)
        eGFR_estimated  — CKD-EPI benadering (gevectoriseerd)
        CaPhos_product  — Calcium × Phosphate
        K_Cr_ratio      — Potassium / Creatinine
    """
    if __has(df, "BUN", "Creatinine"):
        df["BUN_Cr_ratio"] = df["BUN"].div(df["Creatinine"].replace(0, np.nan))

    if __has(df, "Creatinine", "Age", "Gender"):
        cr         = df["Creatinine"]
        age        = df["Age"]
        sex        = df["Gender"]          
        kappa      = np.where(sex == 0, 0.7,    0.9)
        alpha      = np.where(sex == 0, -0.241, -0.302)
        sex_factor = np.where(sex == 0, 1.012,  1.0)
        ratio      = cr / kappa
        exp_val    = np.where(ratio < 1, alpha, -1.200)
        egfr       = 142 * (ratio ** exp_val) * (0.9938 ** age) * sex_factor
        df["eGFR_estimated"] = np.where(
            cr.isna() | age.isna() | (cr <= 0), np.nan, egfr
        )

    if __has(df, "Calcium", "Phosphate"):
        df["CaPhos_product"] = df["Calcium"] * df["Phosphate"]

    if __has(df, "Potassium", "Creatinine"):
        df["K_Cr_ratio"] = df["Potassium"].div(df["Creatinine"].replace(0, np.nan))

    return df

def add_liver_coag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Toegevoegde kolommen:
        Bili_ratio           — Bilirubin_direct / Bilirubin_total
        AST_AlkPhos_ratio    — AST / Alkalinephos (leverpatroon)
        Fibrinogen_Plt_ratio — Fibrinogen / Platelets (DIC-screening)
        Plt_Lactate_ratio    — Platelets / Lactate (gecombineerde index)
        PTT_elevated_flag    — PTT > 60
    """
    if __has(df, "Bilirubin_direct", "Bilirubin_total"):
        df["Bili_ratio"] = df["Bilirubin_direct"].div(
            df["Bilirubin_total"].replace(0, np.nan)
        )

    if __has(df, "AST", "Alkalinephos"):
        df["AST_AlkPhos_ratio"] = df["AST"].div(df["Alkalinephos"].replace(0, np.nan))

    if __has(df, "Fibrinogen", "Platelets"):
        df["Fibrinogen_Plt_ratio"] = df["Fibrinogen"].div(
            df["Platelets"].replace(0, np.nan)
        )

    if __has(df, "Platelets", "Lactate"):
        df["Plt_Lactate_ratio"] = df["Platelets"].div(df["Lactate"].replace(0, np.nan))

    if "PTT" in df.columns:
        df["PTT_elevated_flag"] = (df["PTT"] > 60).astype(float)

    return df

def add_hematology_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Toegevoegde kolommen:
        MCHC_approx                  — Hgb / Hct × 100
        Hgb_Age_ratio                — Hgb / Age
        SIRS_WBC_Temp                — WBC afwijkend ÉN temp afwijkend (0/1)
        Thrombocytopenia_flag        — Platelets < 100
        Severe_Thrombocytopenia_flag — Platelets < 50
    """
    if __has(df, "Hgb", "Hct"):
        df["MCHC_approx"] = df["Hgb"].div(df["Hct"].replace(0, np.nan)) * 100

    if __has(df, "Hgb", "Age"):
        df["Hgb_Age_ratio"] = df["Hgb"].div(df["Age"].replace(0, np.nan))

    if __has(df, "WBC", "Temp"):
        wbc_abnormal  = ((df["WBC"]  < 4)  | (df["WBC"]  > 12)).astype(float)
        temp_abnormal = ((df["Temp"] < 36) | (df["Temp"] > 38)).astype(float)
        df["SIRS_WBC_Temp"] = wbc_abnormal * temp_abnormal

    if "Platelets" in df.columns:
        df["Thrombocytopenia_flag"]        = (df["Platelets"] < 100).astype(float)
        df["Severe_Thrombocytopenia_flag"] = (df["Platelets"] <  50).astype(float)

    return df

def add_news2_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    National Early Warning Score 2 benadering (zonder GCS en urine-output).

    Toegevoegde kolommen:
        NEWS2_estimated — optelsom van deelscores vitale parameters
    """
    news = pd.Series(0.0, index=df.index)

    if "Resp" in df.columns:
        news += np.select(
            [df["Resp"] <= 8, df["Resp"] <= 11, df["Resp"] <= 20,
             df["Resp"] <= 24, df["Resp"] > 24],
            [3, 1, 0, 2, 3], default=np.nan,
        )

    if "O2Sat" in df.columns:
        news += np.select(
            [df["O2Sat"] <= 91, df["O2Sat"] <= 93,
             df["O2Sat"] <= 95, df["O2Sat"] > 95],
            [3, 2, 1, 0], default=np.nan,
        )

    if "SBP" in df.columns:
        news += np.select(
            [df["SBP"] <= 90, df["SBP"] <= 100, df["SBP"] <= 110,
             df["SBP"] <= 219, df["SBP"] > 219],
            [3, 2, 1, 0, 3], default=np.nan,
        )

    if "HR" in df.columns:
        news += np.select(
            [df["HR"] <= 40, df["HR"] <= 50, df["HR"] <= 90,
             df["HR"] <= 110, df["HR"] <= 130, df["HR"] > 130],
            [3, 1, 0, 1, 2, 3], default=np.nan,
        )

    if "Temp" in df.columns:
        news += np.select(
            [df["Temp"] <= 35.0, df["Temp"] <= 36.0, df["Temp"] <= 38.0,
             df["Temp"] <= 39.0, df["Temp"] > 39.0],
            [3, 1, 0, 1, 2], default=np.nan,
        )

    df["NEWS2_estimated"] = news
    return df

def add_temporal_features(
    df: pd.DataFrame,
    patient_col: str = "Patient_ID",
) -> pd.DataFrame:
    """
    Delta-waarden per patiënt (absolute en procentuele verandering
    t.o.v. vorige meting), gesorteerd op Hour.

    Toegevoegde kolommen per variabele:
        Delta_<var>   — absolute verandering
        PctChg_<var>  — procentuele verandering

    Variabelen: HR, SBP, MAP, Resp, Temp, Lactate, WBC, Creatinine,
                Platelets, ShockIndex, SF_ratio.

    Extra:
        ICU_hours_log — log1p(ICULOS)
    """
    temporal_vars = [
        "HR", "SBP", "MAP", "Resp", "Temp",
        "Lactate", "WBC", "Creatinine", "Platelets",
        "ShockIndex", "SF_ratio",
    ]

    df = df.sort_values([patient_col, "Hour"]).copy()

    for var in temporal_vars:
        if var not in df.columns:
            continue
        df[f"Delta_{var}"]  = df.groupby(patient_col)[var].diff()
        prev                = df.groupby(patient_col)[var].shift(1)
        df[f"PctChg_{var}"] = df[f"Delta_{var}"].div(prev.replace(0, np.nan)) * 100

    if "ICULOS" in df.columns:
        df["ICU_hours_log"] = np.log1p(df["ICULOS"])

    return df

def add_rolling_features(
    df: pd.DataFrame,
    patient_col: str = "Patient_ID",
    windows: list = [3, 6],
) -> pd.DataFrame:
    """
    Rollend gemiddelde en standaarddeviatie per patiënt voor vitale parameters.

    Toegevoegde kolommen per variabele × venster:
        <var>_roll<w>h_mean
        <var>_roll<w>h_std

    Variabelen: HR, SBP, Resp, Temp, O2Sat, Lactate.
    Standaard vensters: 3 en 6 uur.
    """
    rolling_vars = ["HR", "SBP", "Resp", "Temp", "O2Sat", "Lactate"]

    df = df.sort_values([patient_col, "Hour"]).copy()

    for var in rolling_vars:
        if var not in df.columns:
            continue
        for w in windows:
            grp = df.groupby(patient_col)[var]
            df[f"{var}_roll{w}h_mean"] = grp.transform(
                lambda x, w=w: x.rolling(w, min_periods=1).mean()
            )
            df[f"{var}_roll{w}h_std"] = grp.transform(
                lambda x, w=w: x.rolling(w, min_periods=1).std()
            )

    return df

def add_all_features(
    df: pd.DataFrame,
    patient_col: str = "Patient_ID",
    rolling_windows: list = [3, 6],
    include_temporal: bool = True,
    include_rolling: bool = True,
) -> pd.DataFrame:
    """
    Voert alle feature-engineering functies achter elkaar uit en geeft
    het verrijkte DataFrame terug.
 
    Parameters
    ----------
    df               : Ruwe dataframe met ICU-metingen
    patient_col      : Kolomnaam voor patiënt-ID (voor tijdsdynamiek & rolling)
    rolling_windows  : Vensters (uren) voor rolling statistieken
    include_temporal : Of delta-features berekend worden
    include_rolling  : Of rolling features berekend worden (traag op grote datasets)
    """
    df = add_sofa_features(df)
    df = add_hemodynamic_features(df)
    df = add_respiratory_features(df)
    df = add_acid_base_features(df)
    df = add_renal_features(df)
    df = add_liver_coag_features(df)
    df = add_hematology_features(df)
    df = add_news2_score(df)
 
    if include_temporal and patient_col in df.columns:
        df = add_temporal_features(df, patient_col)
 
    if include_rolling and patient_col in df.columns:
        df = add_rolling_features(df, patient_col, rolling_windows)
 
    return df