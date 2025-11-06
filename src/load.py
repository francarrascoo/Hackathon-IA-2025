import pandas as pd
import os

# Constante para todo el proyecto
CODIGO_COMUNA_CONCEPCION = 8101
VIAL_FILE = "data/concepcion_vial.csv"
ACCIDENTES_FILE = "data/accidentes_concepcion.csv"

def load_data():
    """
    Carga la red vial y los accidentes de Concepción.
    Asume que los archivos fueron creados por 'src/targets.py'
    """
    print("src.load: Cargando datos de Concepción...")
    if not os.path.exists(VIAL_FILE) or not os.path.exists(ACCIDENTES_FILE):
        print(f"Error: Archivos no encontrados.")
        print("Por favor, ejecuta 'python src/model.py' primero para simular datos y entrenar.")
        return None, None
    
    try:
        df_vial = pd.read_csv(VIAL_FILE)
        df_accidentes = pd.read_csv(ACCIDENTES_FILE)
        print(f"src.load: Datos cargados: {len(df_vial)} segmentos viales, {len(df_accidentes)} accidentes.")
        return df_vial, df_accidentes
    except Exception as e:
        print(f"Error al cargar archivos: {e}")
        return None, None