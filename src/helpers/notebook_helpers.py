import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scepsis_prediction.evaluation import evaluate_sepsis_score

def export_prediction_set(
        test_patient_ids: np.array, 
        df_original: pd.DataFrame, 
        y_pred: np.ndarray
    ) -> None:

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

def read_dataset(
        path: str = '../../data/train_data.csv', 
        sep: str = ','
    ) -> pd.DataFrame:

    return pd.read_csv(path, sep=sep)
    
def get_gender_percentage(data: pd.DataFrame) -> None: 
    genderLenght = len(data['Gender'])
    gender_vals = data['Gender'].unique()

    def calc_percentage(val: int) -> float:
        return (val /genderLenght) * 100

    for val in gender_vals:
        count = (data['Gender'] == val).sum()
        print(f'{val}: {calc_percentage(count):.2f}%')

def get_sepsis_values(data: pd.DataFrame) -> None:
    if 'SepsisLabel' not in data.columns:
        raise ValueError("Kolom 'SepsisLabel' ontbreekt in de DataFrame")

    counts = data['SepsisLabel'].value_counts().sort_index()
    total = len(data)

    print('Sepsis verdeling:')
    for val, count in counts.items():
        percentage = (count / total) * 100
        print(f'{val}: aantal = {count}, percentage = {percentage:.2f}%')

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

TrainTestSplitOutput = tuple[pd.DataFrame, pd.Series, np.array, pd.DataFrame, pd.Series, np.array]

def get_train_test_data_by_patient(
    data: pd.DataFrame,
    train_patients: pd.DataFrame, 
    test_patients: pd.DataFrame,
    y_target: str = 'Sepsis_Future',
    group_col: str = 'Patient_ID',
    delete_patient_ids: bool = False
) -> TrainTestSplitOutput:
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

def calculate_visit_time(data: pd.DataFrame) -> pd.Series:
    df = data.copy()
    df = df[['Patient_ID', 'Hour']].sort_values(['Patient_ID', 'Hour'])

    visit_duration = (
        df.groupby(['Patient_ID'])['Hour']
        .max()
        .add(1)
    )

    df['visit_duration'] = df.set_index(['Patient_ID']).index.map(visit_duration)

    return df['visit_duration']

def prep_dataset(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()

    # Onnodige kolommen verwijderen
    df = df.drop(columns=['Unnamed: 0'])

    # Sepsis_Future correct instellen
    df = df.sort_values(['Patient_ID', 'Hour'])

    df['Sepsis_Future'] = df.groupby('Patient_ID')['SepsisLabel'].shift(-6)
    df = df.dropna(subset=['Sepsis_Future'])
    
    return df