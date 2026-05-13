import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scepsis_prediction.feature_engineering import add_all_features
from scepsis_prediction.evaluation import evaluate_sepsis_score

TrainTestSplitOutput = tuple[pd.DataFrame, pd.Series, np.array, pd.DataFrame, pd.Series, np.array]

def export_prediction_set(
        test_patient_ids: np.array, 
        df_original: pd.DataFrame, 
        y_pred: np.ndarray
    ) -> None:
    """
    Exporteert de twee csv's die nodig zijn voor het evalueren van utility score. test_patient_ids
    zijn hiervoor nodig om dat deze bij de meeste splits verwijderd worden. Deze moeten dus achteraf terug
    geplakt worden voor de evalutatie.
 
    Parameters
    ----------
    test_patient_ids    : Lijst met de test patients id's
    df_original         : Origniele dataset waar de predictie op geplakt word
    y_pred              : Prediction die het model heeft voorspeld
    """
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
    """
    Deze functie print de utility score het model
 
    Parameters
    ----------
    testset         : Ruwe dataframe met ICU-metingen
    predictions     : Kolomnaam voor patiënt-ID (voor tijdsdynamiek & rolling)
    """    
    utility = evaluate_sepsis_score(testset, predictions)
    print(utility)

def read_dataset(
        path: str = '../../data/train_data.csv', 
        sep: str = ','
    ) -> pd.DataFrame:
    """
    Functie voor het uitlezen van datasets. Deze leest standaard gewoon de train_data uit.
    Kan ook gebruikt worden om andere datasets uit te lezen.
 
    Parameters
    ----------
    path            : Pad naar de dataset
    sep             : Seperator die gebruikt worden in de csv
    """
    return pd.read_csv(path, sep=sep)
    
def get_gender_percentage(data: pd.DataFrame) -> None: 
    """
    Deze functie bererkend de verhouding van genders in de dataset.
 
    Parameters
    ----------
    data            : Dateset waarin de genders staan
    """
    genderLenght = len(data['Gender'])
    gender_vals = data['Gender'].unique()

    def calc_percentage(val: int) -> float:
        return (val /genderLenght) * 100

    for val in gender_vals:
        count = (data['Gender'] == val).sum()
        print(f'{val}: {calc_percentage(count):.2f}%')

def get_sepsis_values(data: pd.DataFrame) -> None:
    """
    Funtie die berekent hoeveel procent van de dataset wel/geen sepis heeft
    op basis van de sepsisLabels.
 
    Parameters
    ----------
    data            : Dateset waarin de sepsislabels hebben
    """
    if 'SepsisLabel' not in data.columns:
        raise ValueError("Kolom 'SepsisLabel' ontbreekt in de DataFrame")

    counts = data['SepsisLabel'].value_counts().sort_index()
    total = len(data)

    print('Sepsis verdeling:')
    for val, count in counts.items():
        percentage = (count / total) * 100
        print(f'{val}: aantal = {count}, percentage = {percentage:.2f}%')

def train_test_split_by_patient(
    original_df: pd.DataFrame, 
    group_col: str = 'Patient_ID', 
    test_size: int = 0.2, 
    random_state: int = 42
):
    """
    Voert alle feature-engineering functies achter elkaar uit en geeft
    het verrijkte DataFrame terug.
 
    Parameters
    ----------
    original_df     : Orignele dataset waaruit alle unieke patient id's worden gehaald
    group_col       : De kolom waarop gegroupeerd word
    test_size       : Standaard test size, kan aangepast worden indien mogelijk
    random_state    : Standaard random state, kan aangepast worden indien mogelijk
    """
    df = original_df.copy()
    patients = df[group_col].unique()

    train_patients, test_patients = train_test_split(
        patients, test_size=test_size, random_state=random_state
    )

    return train_patients, test_patients

def get_train_test_data_by_patient(
    original_df: pd.DataFrame,
    train_patients: pd.DataFrame, 
    test_patients: pd.DataFrame,
    y_target: str = None,
    delete_patient_ids: bool = False
) -> TrainTestSplitOutput:
    """
    Voert train test split uit op basis van de patient id's om dataleaks te verkomen
 
    Parameters
    ----------
    original_df         : Orginele dataset. Hieruit worden allemaal kolommen verwijderd mits 
                          ze aanwezig zijn in de datset. Deze zijn niet nodig voor het trainen
                          van het model
    train_patients      : De train set met patient id's
    test_patients       : De test set met patient id's
    y_target            : Target waarop de dataset getraind word. Dit was voorheen Sepsis_Future, nu 
                          gewoon SepsisLabel
    delete_patient_ids  : Of de patients id's verwijderd moeten worden na 
    """
    df = original_df.copy()

    qsofa_features = [col for col in df.columns if 'qsofa' in col.lower()]

    sofa_features = [
        col for col in df.columns
        if 'sofa' in col.lower() and col not in qsofa_features
    ]

    if 'SF_ratio' in df.columns:
        sofa_features.append('SF_ratio')

    unit_features = [col for col in df.columns if 'unit' in col.lower()]

    leak_cols = ['SepsisLabel', 'Sepsis_Future']

    drop_cols = [
        *qsofa_features, 
        *sofa_features, 
        *unit_features
    ]

    train = train_patients.copy()
    test = test_patients.copy()

    train_df = df[df['Patient_ID'].isin(train)]
    test_df = df[df['Patient_ID'].isin(test)]

    y_train = train_df[y_target].astype(int)
    X_train = train_df.drop(columns=[c for c in drop_cols + leak_cols if c in train_df.columns])

    y_test = test_df[y_target].astype(int)
    X_test = test_df.drop(columns=[c for c in drop_cols + leak_cols if c in test_df.columns])

    # Patient_ID is nodig voor het bereken van de utility score, maar het model mag niet trainen hierop
    # Daarom worden deze eruit gefilterd en daarna verwijderd uit de test sets
    train_patient_ids = X_train['Patient_ID'].values
    test_patient_ids = X_test['Patient_ID'].values

    if delete_patient_ids:
        X_train = X_train.drop(columns=['Patient_ID'])
        X_test = X_test.drop(columns=['Patient_ID'])

    return X_train, y_train, train_patient_ids, X_test, y_test, test_patient_ids

def calculate_visit_time(data: pd.DataFrame) -> pd.Series:
    """
    Voert alle feature-engineering functies achter elkaar uit en geeft
    het verrijkte DataFrame terug.
 
    Parameters
    ----------
    data               : Set waarvan de visit_duration berekend word
    """    
    df = data.copy()
    df = df[['Patient_ID', 'Hour']].sort_values(['Patient_ID', 'Hour'])

    visit_duration = (
        df.groupby(['Patient_ID'])['Hour']
        .max()
        .add(1)
    )

    df['visit_duration'] = df.set_index(['Patient_ID']).index.map(visit_duration)

    return df['visit_duration']

def prep_dataset(
        data: pd.DataFrame,
        add_sepsis_future: bool = True
    ) -> pd.DataFrame:
    """
    Prepereerd de dataset als er nieuwe schone set nodig voor 
    bijvoorbeeld een nieuwe cycle.
 
    Parameters
    ----------
    data                : Dataset die voorbereid word
    add_sepsis_future   : Bool om dit uit te voeren of niet. Op een gegevenen moment is bepaald dat dit 
                          niet meer nodig was. Dit is nog wel nodig voor eerder uitgevoerde cycles.
    """
    
    df = data.copy()

    # Onnodige kolommen verwijderen
    df = df.drop(columns=['Unnamed: 0'])

    # Sepsis_Future correct instellen
    df = df.sort_values(['Patient_ID', 'Hour'])

    if add_sepsis_future:
        df['Sepsis_Future'] = df.groupby('Patient_ID')['SepsisLabel'].shift(-6)
        df = df.dropna(subset=['Sepsis_Future'])
    
    return df

# def get_prepped_dataset(
#         delete_patient_ids: bool = True,
#         include_temporal_values: bool = False,
#         include_rolling_values: bool = False,
#         add_sepsis_future: bool = False,
#         y_target: str = None
# ) -> pd.DataFrame:
#     """
#     Combinatie van alle voorgaande functies samengevoegd tot een functie. Deze word voornamelijk gebruikt bij het laden
#     van modellen die gegeneerd zijn door optuna. Dit zin de basis stappen die doorlopen worden voor iedere dataset voordat
#     de modellen daarop getraind worden. 

#     Omdat het alle functies hiervoor samengevoegd zijn worden de parameters ook niet toegelicht.
 
#     """

#     df = read_dataset()
#     df = prep_dataset(df, add_sepsis_future)

#     df = add_all_features(
#         df=df, 
#         include_temporal=include_temporal_values, 
#         include_rolling=include_rolling_values
#     )

#     df = df.ffill()
#     df = df.dropna()

#     train_patients, test_patients = train_test_split_by_patient(df)
#     _, _, _, X_test, _, test_patient_ids = get_train_test_data_by_patient(
#         df,
#         train_patients,
#         test_patients,
#         y_target,
#         delete_patient_ids
#     )

#     return df, X_test, test_patient_ids

def run_model(path: str, X_test: pd.DataFrame, threshold: float = 0.5):
    """
    Runned het model gemaakt door optuna aan de hand van het 
    optimalistatie script.
 
    Parameters
    ----------
    path           : Pad van het model
    X_test          : X_test dataset die nodig voor het voorspellen van de y_pred
    threshold       : Threshold meegegeven uit model
    """
    loaded = joblib.load(path)
    threshold = loaded.get("threshold", 0.5)

    proba = loaded.predict_proba(X_test)[:, 1]
    y_pred = (proba >= threshold).astype(int)

    return y_pred