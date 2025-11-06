# src/features.py
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

def get_preprocessor():
    """
    Retorna el ColumnTransformer para preprocesar las features.
    (Incluye 'Comuna' como feature)
    """
    
    categorical_features = ['Comuna', 'type', 'hora_del_dia', 'dia_semana', 'mes']
    numeric_features = ['Shape__Length']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='drop', # Ignora 'FID' y 'Year' si aún estuvieran
        verbose_feature_names_out=False
    )
    return preprocessor