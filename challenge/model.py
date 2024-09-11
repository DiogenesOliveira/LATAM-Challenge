from datetime import datetime
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import pickle
from xgboost import XGBClassifier

PICKLE_MODEL_PATH = "data/model.pkl"
FULL_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
EXPECTED_FEATURES = [
            "OPERA_Latin American Wings", 
            "MES_7",
            "MES_10",
            "OPERA_Grupo LATAM",
            "MES_12",
            "TIPOVUELO_I",
            "MES_4",
            "MES_11",
            "OPERA_Sky Airline",
            "OPERA_Copa Air"
        ]

class DelayModel:
    def __init__(self):
        self._model = self._load_pickle_model(PICKLE_MODEL_PATH)

    def _load_pickle_model(self, path: str) -> Union[None, object]:
        with open(path, "rb") as f:
            model = pickle.load(f)
            return model
    
    def _is_delayed(self, data: pd.DataFrame) -> pd.DataFrame:
        data["min_diff"] = data.apply(self._get_min_diff, axis=1)
        data["delay"] = np.where(data["min_diff"] > 15, 1, 0)
        return data

    def preprocess(self, data: pd.DataFrame, target_column: str = None) -> Union[Tuple[pd.DataFrame, pd.Series], pd.DataFrame]:
        features = pd.concat([
            pd.get_dummies(data['OPERA'], prefix='OPERA'),
            pd.get_dummies(data['TIPOVUELO'], prefix='TIPOVUELO'),
            pd.get_dummies(data['MES'], prefix='MES')
        ], axis=1)

        for col in EXPECTED_FEATURES:
            if col not in features.columns:
                features[col] = 0

        features = features[EXPECTED_FEATURES]

        if target_column:
            data = self._is_delayed(data)
            target = data[target_column]
            return features, target
        else:
            return features

    def _get_min_diff(self, row) -> float:
        fecha_o = datetime.strptime(row['Fecha-O'], FULL_DATE_FORMAT)
        fecha_i = datetime.strptime(row['Fecha-I'], FULL_DATE_FORMAT)
        return (fecha_o - fecha_i).total_seconds() / 60

    def fit(self, features: pd.DataFrame, target: pd.Series) -> None:
        n_y0 = len(target[target == 0])
        n_y1 = len(target[target == 1])
        scale = n_y0 / n_y1

        self._model = XGBClassifier(random_state=1, learning_rate=0.01, scale_pos_weight=scale)
        self._model.fit(features, target)
        with open(PICKLE_MODEL_PATH, "wb") as f:
            pickle.dump(self._model, f)

    def predict(self, features: pd.DataFrame) -> List[int]:
        predictions = self._model.predict(features)
        return predictions.tolist()
