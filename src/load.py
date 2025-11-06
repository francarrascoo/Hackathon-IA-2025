# src/load.py
import pandas as pd
import os

# --- NOMBRES DE ARCHIVOS PROCESADOS (V15) ---
# Archivos creados por src/targets.py y usados por src/model.py
PROCESSED_TRAIN_FILE = "data/train_dataset_full.csv"
PROCESSED_TEST_FILE = "data/test_dataset_full.csv"

# Archivos de "producción" usados por api/main.py y src/rag.py
# (El vial procesado de toda la región)
VIAL_FILE = "data/biobio_vial_procesado.csv"
# (El RAG usa todos los años, pero la API usa el más reciente para el dashboard)
SINIESTROS_2024_FILE = "data/Siniestros_2024.csv"

def load_train_test_data():
    """
    Carga los datasets de ENTRENAMIENTO (2021-23) y PRUEBA (2024)
    pre-procesados por src/targets.py.
    """
    print(f"src.load: Cargando datos procesados de train/test...")
    
    if not os.path.exists(PROCESSED_TRAIN_FILE) or not os.path.exists(PROCESSED_TEST_FILE):
        print(f"Error: Archivos procesados '{PROCESSED_TRAIN_FILE}' o '{PROCESSED_TEST_FILE}' no encontrados.")
        print("Por favor, ejecuta 'python -m src.model' primero para generar estos archivos.")
        return None, None
    
    try:
        df_train = pd.read_csv(PROCESSED_TRAIN_FILE)
        df_test = pd.read_csv(PROCESSED_TEST_FILE)
        print(f"src.load: Datos cargados: {len(df_train)} muestras de entreno, {len(df_test)} muestras de prueba.")
        return df_train, df_test
    
    except Exception as e:
        print(f"Error al cargar archivos procesados: {e}")
        return None, None