import pandas as pd
import pickle
import numpy as np
from typing import Tuple, Union, List
from xgboost import XGBClassifier
from datetime import datetime

PICKLE_MODEL_PATH = "data/model.pkl"
FULL_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
DAY_MONTH_FORMAT = '%d-%b'
HOUR_FORMAT = "%H:%M"

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
        """
        Prepare raw data for training or prediction.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.Series]: features and target.
            or
            pd.DataFrame: features.
        """

        # Selecting features
        features = pd.concat([
            pd.get_dummies(data['OPERA'], prefix='OPERA'),
            pd.get_dummies(data['TIPOVUELO'], prefix='TIPOVUELO'),
            pd.get_dummies(data['MES'], prefix='MES')
        ], axis=1)

        # Ensure columns match expected features
        expected_features = [
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
        # Add missing columns with zeros
        for col in expected_features:
            if col not in features.columns:
                features[col] = 0

        # Select and order the columns to match expected features
        features = features[expected_features]

        if target_column:
            data = self._is_delayed(data)
            target = data[target_column]
            return features, target
        else:
            return features

    def _get_period_day(self, date: str) -> str:
        date_time = datetime.strptime(date, FULL_DATE_FORMAT).time()
        morning_min = datetime.strptime("05:00", HOUR_FORMAT).time()
        morning_max = datetime.strptime("11:59", HOUR_FORMAT).time()
        afternoon_min = datetime.strptime("12:00", HOUR_FORMAT).time()
        afternoon_max = datetime.strptime("18:59", HOUR_FORMAT).time()

        if morning_min <= date_time <= morning_max:
            return 'mañana'
        elif afternoon_min <= date_time <= afternoon_max:
            return 'tarde'
        else:
            return 'noche'

    def _is_high_season(self, fecha: str) -> int:
        fecha_año = int(fecha.split('-')[0])
        fecha = datetime.strptime(fecha, FULL_DATE_FORMAT)
        range1_min = datetime.strptime('15-Dec', DAY_MONTH_FORMAT).replace(year=fecha_año)
        range1_max = datetime.strptime('31-Dec', DAY_MONTH_FORMAT).replace(year=fecha_año)
        range2_min = datetime.strptime('1-Jan', DAY_MONTH_FORMAT).replace(year=fecha_año)
        range2_max = datetime.strptime('3-Mar', DAY_MONTH_FORMAT).replace(year=fecha_año)
        range3_min = datetime.strptime('15-Jul', DAY_MONTH_FORMAT).replace(year=fecha_año)
        range3_max = datetime.strptime('31-Jul', DAY_MONTH_FORMAT).replace(year=fecha_año)
        range4_min = datetime.strptime('11-Sep', DAY_MONTH_FORMAT).replace(year=fecha_año)
        range4_max = datetime.strptime('30-Sep', DAY_MONTH_FORMAT).replace(year=fecha_año)

        if ((range1_min <= fecha <= range1_max) or 
            (range2_min <= fecha <= range2_max) or 
            (range3_min <= fecha <= range3_max) or
            (range4_min <= fecha <= range4_max)):
            return 1
        else:
            return 0

    def _get_min_diff(self, row) -> float:
        fecha_o = datetime.strptime(row['Fecha-O'], FULL_DATE_FORMAT)
        fecha_i = datetime.strptime(row['Fecha-I'], FULL_DATE_FORMAT)
        return (fecha_o - fecha_i).total_seconds() / 60

    def fit(self, features: pd.DataFrame, target: pd.Series) -> None:
        # Balance classes
        n_y0 = len(target[target == 0])
        n_y1 = len(target[target == 1])
        scale = n_y0 / n_y1

        # Model training
        self._model = XGBClassifier(random_state=1, learning_rate=0.01, scale_pos_weight=scale)
        self._model.fit(features, target)
        with open(PICKLE_MODEL_PATH, "wb") as f:
            pickle.dump(self._model, f)

    def predict(self, features: pd.DataFrame) -> List[int]:
        if self._model is None:
            raise ValueError("Model has not been trained. Call fit before predict.")
        
        predictions = self._model.predict(features)
        return predictions.tolist()
