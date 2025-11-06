import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

def create_feature_dataset(df_target_data, df_vial):
    """
    Combina los datos de target (muestras) con los features de la red vial[cite: 142].
    """
    print("src.features: Combinando etiquetas con features viales...")
    
    df_vial_features = df_vial[['FID', 'type', 'Shape__Length', 'comuna']]
    
    # Merge
    df_dataset = df_target_data.merge(
        df_vial_features, 
        left_on='FID_via', 
        right_on='FID', 
        how='left'
    )
    
    # Limpieza
    df_dataset = df_dataset.drop_duplicates()
    df_dataset = df_dataset.dropna(subset=['type', 'Shape__Length'])
    df_dataset = df_dataset.drop(['FID_via', 'FID'], axis=1)
    
    print(f"src.features: Dataset de features creado con {len(df_dataset)} filas.")
    return df_dataset

def get_preprocessor():
    """
    Retorna el ColumnTransformer para preprocesar las features.
    """
    
    # Features categóricas y numéricas
    categorical_features = ['type', 'comuna', 'hora_del_dia', 'dia_semana', 'mes']
    numeric_features = ['Shape__Length']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='drop',
        verbose_feature_names_out=False
    )
    return preprocessor