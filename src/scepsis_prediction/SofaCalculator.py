import pandas as pd
import numpy as np

### Deze klasse is gabaseerd op de `test_data.csv` dataset.
### Hierin staan niet alle waardes die nodig zijn om een Quick SOFA en SOFA te berekenen.
### De berekeningen worden gedaan op basis van de waardes die wel in de dataset staan.

class SofaCalculator:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.redundant_qsofa_colls = ['qsofa_resp', 'qsofa_sbp']
        self.redundant_sofa_colls = ['sofa_resp', 'sofa_coag', 'sofa_liver', 'sofa_cv', 'sofa_renal', 'SF_ratio']

    def calculate_qsofa(self, returnColumns: bool = False) -> pd.DataFrame:
        self.df = self.__calculate_qsofa(self.df.copy())

        if returnColumns:
            return self.df
        else:
            return self.df.drop(columns=self.redundant_qsofa_colls)

    
    def calculate_sofa(self, returnColumns: bool = False) -> pd.DataFrame:
        self.df = self.__calculate_sofa(self.df.copy())

        if returnColumns:
            return self.df
        else:
            return self.df.drop(columns=self.redundant_sofa_colls)
    
    def calculate_all_values(
        self,
        returnQSofaColumns: bool = False,
        returnSofaColumns: bool = False
    ) -> pd.DataFrame:
        self.df = self.__calculate_qsofa(self.df.copy())
        self.df = self.__calculate_sofa(self.df.copy())

        if returnQSofaColumns and returnSofaColumns:
            return self.df
        elif returnQSofaColumns:
            return self.df.drop(columns=self.redundant_sofa_colls)
        elif returnSofaColumns:
            return self.df.drop(columns=self.redundant_qsofa_colls)
        else:
            redundant_colls = [*self.redundant_sofa_colls, *self.redundant_qsofa_colls]
            return self.df.drop(columns=redundant_colls)


    def hasSepsis(self) -> pd.Series:
        required_cols = ["qSOFA_partial", "SOFA_modified_total"]

        if all(col in self.df.columns for col in required_cols):
            return (
                (self.df["qSOFA_partial"] >= 2) &
                (self.df["SOFA_modified_total"] >= 2)
            )
        else:
            raise IndexError("Kolom of kolommen bestaan niet")
        

    # Worker methods
    def __calculate_qsofa(self, df: pd.DataFrame) -> pd.DataFrame:
        df["qsofa_resp"] = (df["Resp"] >= 22).astype(int)
        df["qsofa_sbp"] = (df["SBP"] <= 100).astype(int)

        df["qSOFA_partial"] = (
            df["qsofa_resp"] +
            df["qsofa_sbp"]
        )

        return df

    def __calculate_sofa(self, df: pd.DataFrame) -> pd.DataFrame:
        spo2_col = self.__get_spo2_column(df)
        df["SF_ratio"] = df[spo2_col] / df["FiO2"]

        df["sofa_resp"] = df["SF_ratio"].apply(self.__resp_score)
        df["sofa_coag"] = df["Platelets"].apply(self.__coag_score)
        df["sofa_liver"] = df["Bilirubin_total"].apply(self.__liver_score)
        df["sofa_cv"] = df["MAP"].apply(self.__cv_score)
        df["sofa_renal"] = df["Creatinine"].apply(self.__renal_score)

        score_cols = [
            "sofa_resp",
            "sofa_coag",
            "sofa_liver",
            "sofa_cv",
            "sofa_renal"
        ]

        df["SOFA_modified_total"] = df[score_cols].sum(axis=1)

        return df

    # Helpers
    def __get_spo2_column(self, df):
        return "O2Sat" if "O2Sat" in df.columns else "SaO2"
    
    def __resp_score(self, sf):
        if pd.isna(sf):
            return np.nan
        elif sf >= 400:
            return 0
        elif sf >= 300:
            return 1
        elif sf >= 200:
            return 2
        elif sf >= 100:
            return 3
        return 4
    
    def __coag_score(self, plt):
        if pd.isna(plt):
            return np.nan
        elif plt >= 150:
            return 0
        elif plt >= 100:
            return 1
        elif plt >= 50:
            return 2
        elif plt >= 20:
            return 3
        return 4
    
    def __liver_score(self, bili):
        if pd.isna(bili):
            return np.nan
        elif bili < 1.2:
            return 0
        elif bili < 2.0:
            return 1
        elif bili < 6.0:
            return 2
        elif bili < 12.0:
            return 3
        return 4

    def __cv_score(self, map_val):
        if pd.isna(map_val):
            return np.nan
        return 0 if map_val >= 70 else 1

    def __renal_score(self, creat):
        if pd.isna(creat):
            return np.nan
        elif creat < 1.2:
            return 0
        elif creat < 2.0:
            return 1
        elif creat < 3.5:
            return 2
        elif creat < 5.0:
            return 3
        return 4