import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

# --- Configuración ---
DATA_FILES = [
    'Siniestros_2021.csv',
    'Siniestros_2022.csv',
    'Siniestros_2023.csv',
    'Siniestros_2024.csv'
]
ARTIFACTS_DIR = 'artifacts'
MODEL_PATH = os.path.join(ARTIFACTS_DIR, 'street_risk_model.joblib')
LABEL_ENCODER_PATH = os.path.join(ARTIFACTS_DIR, 'label_encoder.joblib')
MODEL_COLUMNS_PATH = os.path.join(ARTIFACTS_DIR, 'model_columns.joblib')

# --- MAPA DE COLUMNAS PARA ESTANDARIZAR ---
COLUMN_MAP = {
    'Fecha': 'fecha',
    'COMUNA': 'comuna',
    'Comuna': 'comuna',
    'CALLE_UNO': 'calle_uno',
    'Calle_Uno': 'calle_uno',
    'Tipo_Accid': 'tipo_accid'
}

def load_data(files):
    """Carga, estandariza y concatena múltiples archivos CSV."""
    df_list = []
    for file in files:
        try:
            path = os.path.join('data', file)
            df = pd.read_csv(path)
            
            df = df.rename(columns=COLUMN_MAP)
            
            if 'fecha' in df.columns:
                df_list.append(df)
            else:
                print(f"Advertencia: 'fecha' no encontrada en {path} (después de renombrar)")
        except FileNotFoundError:
            print(f"Error: Archivo no encontrado {path}")
            return None
        except Exception as e:
            print(f"Error cargando {path}: {e}")
            return None
            
    if not df_list:
        print("Error: No se pudieron cargar datos.")
        return None
        
    full_df = pd.concat(df_list, ignore_index=True)
    
    try:
        full_df['fecha'] = pd.to_datetime(full_df['fecha'])
    except Exception as e:
        print(f"Error al parsear 'fecha': {e}. Usando 'errors=coerce'")
        full_df['fecha'] = pd.to_datetime(full_df['fecha'], errors='coerce')

    columnas_requeridas = ['fecha', 'comuna', 'calle_uno', 'tipo_accid']
    full_df = full_df.dropna(subset=columnas_requeridas)
    
    full_df['year'] = full_df['fecha'].dt.year
    return full_df


def create_features_by_location(df):
    """
    Agrega los datos por ubicación (Comuna, Calle) y crea features 
    para el modelo de ML, usando nombres estándar en minúsculas.
    """
    if df.empty:
        return pd.DataFrame()

    # --- INICIO DE LA CORRECCIÓN V3 (MÁS ROBUSTA) ---
    df['tipo_accid'] = df['tipo_accid'].astype(str)

    agg_ops = {
        'fecha': 'count',
        'tipo_accid': [
            ('num_colision', lambda x: (x.str.contains('COLIS', case=False, na=False)).sum()),
            ('num_atropello', lambda x: (x.str.contains('ATROP', case=False, na=False)).sum()),
            ('num_caida', lambda x: (x.str.contains('CAIDA', case=False, na=False)).sum()),
        ]
    }
    # --- FIN DE LA CORRECCIÓN V3 ---
    
    location_df = df.groupby(['comuna', 'calle_uno']).agg(agg_ops)
    
    location_df.columns = ['_'.join(col).strip() for col in location_df.columns.values]
    location_df = location_df.rename(columns={'fecha_count': 'total_accidents'})
    
    location_df = location_df.rename(columns={
        'tipo_accid_num_colision': 'num_colision',
        'tipo_accid_num_atropello': 'num_atropello',
        'tipo_accid_num_caida': 'num_caida'
    })

    total_valid_accidents = location_df['total_accidents'].replace(0, 1)

    location_df['perc_colision'] = location_df['num_colision'] / total_valid_accidents
    location_df['perc_atropello'] = location_df['num_atropello'] / total_valid_accidents
    location_df['perc_caida'] = location_df['num_caida'] / total_valid_accidents
    
    bins = [-1, 5, 15, float('inf')]
    labels = ['Bajo', 'Medio', 'Alto']
    location_df['risk_label'] = pd.cut(location_df['total_accidents'], bins=bins, labels=labels)
    
    return location_df

def main():
    """Función principal para entrenar y guardar el modelo."""
    print("Iniciando el script de entrenamiento...")
    
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    
    # 1. Cargar datos
    full_df = load_data(DATA_FILES)
    if full_df is None:
        return

    print(f"Datos cargados: {len(full_df)} siniestros totales.")

    # 2. Validación Temporal
    print("Preparando conjuntos de validación temporal...")
    train_data = full_df[full_df['year'].isin([2021, 2022, 2023])]
    test_data = full_df[full_df['year'] == 2024]

    print(f"Filas cargadas para ENTRENAMIENTO (2021-2023): {len(train_data)}")
    print(f"Filas cargadas para PRUEBA (2024): {len(test_data)}")

    # 3. Ingeniería de Features
    train_locations = create_features_by_location(train_data)
    test_locations = create_features_by_location(test_data)

    if train_locations.empty or test_locations.empty:
        print("Error: No se pudieron crear features. Data de entrenamiento o prueba vacía.")
        return

    # 4. Definir Features (X) y Target (y)
    feature_cols = ['perc_colision', 'perc_atropello', 'perc_caida']
    target_col = 'risk_label'

    # Unir todos los índices (calles) de ambos periodos
    all_locations_index = train_locations.index.union(test_locations.index)
    
    # --- INICIO DE LA CORRECCIÓN ---
    # 1. Separar X (features) e y (target) ANTES de reindexar
    X_train_df = train_locations[feature_cols]
    y_train_series = train_locations[target_col]
    
    X_test_df = test_locations[feature_cols]
    y_test_series = test_locations[target_col]

    # 2. Reindexar X (features numéricas) con fill_value=0
    X_train = X_train_df.reindex(all_locations_index, fill_value=0)
    X_test = X_test_df.reindex(all_locations_index, fill_value=0)

    # 3. Reindexar y (target categórico) con fill_value='Bajo'
    y_train = y_train_series.reindex(all_locations_index, fill_value='Bajo')
    y_test = y_test_series.reindex(all_locations_index, fill_value='Bajo')
    # --- FIN DE LA CORRECCIÓN ---

    # Codificar etiquetas (y) a números
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)

    print(f"Clases del modelo: {le.classes_}")

    # 5. Entrenar modelo de validación
    print("Entrenando modelo (Random Forest) para validación...")
    validation_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    validation_model.fit(X_train, y_train_encoded)

    # 6. Evaluar en el conjunto de prueba temporal (datos 2024)
    print("\n--- Reporte de Validación (Prediciendo 2024) ---")
    y_pred = validation_model.predict(X_test)
    print(classification_report(y_test_encoded, y_pred, target_names=le.classes_))
    print("--------------------------------------------------\n")

    # 7. Entrenar Modelo de Producción (con TODOS los datos)
    print("Entrenando modelo de producción final con TODOS los datos (2021-2024)...")
    
    all_data_locations = create_features_by_location(full_df)

    # --- CORRECCIÓN DE PRODUCCIÓN ---
    # 1. Separar X e y de producción
    X_prod_df = all_data_locations[feature_cols]
    y_prod_series = all_data_locations[target_col]

    # 2. Alinear X e y de producción al índice completo (de train+test)
    X_prod_aligned = X_prod_df.reindex(all_locations_index, fill_value=0)
    y_prod_aligned = y_prod_series.reindex(all_locations_index, fill_value='Bajo')
    
    # 3. Codificar 'y' de producción
    y_prod_encoded = le.transform(y_prod_aligned)
    # --- FIN CORRECCIÓN DE PRODUCCIÓN ---

    production_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    production_model.fit(X_prod_aligned, y_prod_encoded)

    # 8. Guardar Artefactos
    print(f"Guardando artefactos en {ARTIFACTS_DIR}...")
    
    joblib.dump(production_model, MODEL_PATH)
    joblib.dump(le, LABEL_ENCODER_PATH)
    joblib.dump(feature_cols, MODEL_COLUMNS_PATH)

    print("\n¡Entrenamiento completado!")
    print(f"Modelo guardado en: {MODEL_PATH}")
    print(f"Codificador de etiquetas guardado en: {LABEL_ENCODER_PATH}")
    print(f"Columnas del modelo guardadas en: {MODEL_COLUMNS_PATH}")

if __name__ == "__main__":
    main()