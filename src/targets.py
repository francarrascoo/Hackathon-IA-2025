import pandas as pd
import numpy as np
import random
import os
from src.load import CODIGO_COMUNA_CONCEPCION, VIAL_FILE, ACCIDENTES_FILE

# CONSTANTES DE DATOS REALES
NUEVO_CSV_SINIESTROS = "data/Siniestros_urbanos_biobio_2024.csv"
COMUNAS_OBJETIVO = ["CONCEPCION", "SAN PEDRO DE LA PAZ"]

def _prepare_real_data():
    """
    Función interna para:
    1. Filtrar la Red Vial (Red_Vial_de_Chile.csv) por nuestras comunas.
    2. Cargar los Siniestros Reales (Siniestros_urbanos_biobio_2024.csv).
    3. Cruzar Siniestros con Red Vial por nombre de calle para asignar un FID a cada accidente.
    4. Guardar los accidentes procesados.
    """
    print("src.targets: Procesando datos reales (Siniestros + Red Vial)...")
    
    # 1. Cargar y filtrar Red Vial
    try:
        df_vial_full = pd.read_csv("data/Red_Vial_de_Chile.csv")
    except FileNotFoundError:
        print("Error: 'data/Red_Vial_de_Chile.csv' no encontrado.")
        return
    
    df_vial = df_vial_full[df_vial_full['nombre_com'].isin(COMUNAS_OBJETIVO)].copy()
    if df_vial.empty:
        print(f"Error: No se encontraron datos viales para {COMUNAS_OBJETIVO}.")
        return
    
    df_vial['calle_join_key'] = df_vial['nombre'].str.upper().str.strip()
    df_vial.to_csv(VIAL_FILE, index=False)
    print(f"src.targets: Red vial filtrada guardada en '{VIAL_FILE}'.")

    # 2. Cargar Siniestros Reales
    try:
        df_siniestros = pd.read_csv(NUEVO_CSV_SINIESTROS)
    except FileNotFoundError:
        print(f"Error: '{NUEVO_CSV_SINIESTROS}' no encontrado. Asegúrate de que esté en /data.")
        return
    
    df_siniestros = df_siniestros[df_siniestros['Comuna'].isin(COMUNAS_OBJETIVO)]
    print(f"src.targets: {len(df_siniestros)} siniestros reales cargados para {COMUNAS_OBJETIVO}.")
    df_siniestros['calle_join_key'] = df_siniestros['Calle_Uno'].str.upper().str.strip()

    # 3. Cruzar Siniestros con Red Vial (Join por nombre de calle y comuna)
    df_accidentes_procesados = pd.merge(
        df_siniestros,
        df_vial[['FID', 'calle_join_key', 'nombre_com']], # 'FID' se convertirá en 'FID_y'
        how="inner",
        left_on=['calle_join_key', 'Comuna'],
        right_on=['calle_join_key', 'nombre_com']
    )
    
    # Renombramos 'FID_y' (el FID de la red vial) a 'FID_via'
    df_accidentes_procesados = df_accidentes_procesados.rename(columns={"FID_y": "FID_via"})

    # 4. Guardar los accidentes con FID
    df_accidentes_procesados.to_csv(ACCIDENTES_FILE, index=False)
    print(f"src.targets: Procesamiento completo. {len(df_accidentes_procesados)} accidentes fueron exitosamente mapeados a un FID vial.")
    print(f"Archivo de accidentes procesados guardado en '{ACCIDENTES_FILE}'.")


def get_target_data(df_vial, df_accidentes):
    """
    Crea el set de datos para entrenamiento (target=1 vs target=0).
    """
    print("src.targets: Creando etiquetas (positivas y negativas)...")
    
    # 1. Etiquetas Positivas (Accidentes REALES)
    df_acc = df_accidentes.copy()
    
    if df_acc.empty or 'FID_via' not in df_acc.columns:
        print("src.targets: ADVERTENCIA: No se encontraron accidentes positivos (reales) tras el cruce con la red vial.")
        print("src.targets: El modelo se entrenará solo con muestras negativas (riesgo bajo).")
        df_acc_pos = pd.DataFrame(columns=['FID_via', 'hora_del_dia', 'dia_semana', 'mes', 'target'])
    else:
        print(f"src.targets: Creando {len(df_acc)} etiquetas positivas (reales).")
        df_acc['timestamp'] = pd.to_datetime(df_acc['Fecha']) # Usar la fecha real
        df_acc['hora_del_dia'] = df_acc['timestamp'].dt.hour
        df_acc['dia_semana'] = df_acc['timestamp'].dt.dayofweek
        df_acc['mes'] = df_acc['timestamp'].dt.month
        df_acc['target'] = 1
        df_acc_pos = df_acc[['FID_via', 'hora_del_dia', 'dia_semana', 'mes', 'target']]

    # 2. Etiquetas Negativas (No-Accidentes)
    num_neg_samples = max(100, len(df_acc_pos) * 2) 
    neg_samples = []
    for _ in range(num_neg_samples):
        neg_samples.append({
            "FID_via": random.choice(df_vial['FID'].values),
            "hora_del_dia": random.randint(0, 23),
            "dia_semana": random.randint(0, 6),
            "mes": random.randint(1, 12),
            "target": 0
        })
    df_neg = pd.DataFrame(neg_samples)

    # 3. Dataset final
    df_target_data = pd.concat([df_acc_pos, df_neg], ignore_index=True)
    
    # --- INICIO DE LA CORRECCIÓN (V5) ---
    # El concat de un DF vacío (object) y un DF con ints (int) 
    # puede resultar en un dtype 'object'.
    # Forzamos que 'target' sea numérico y llenamos NaNs (con 0).
    df_target_data['target'] = pd.to_numeric(df_target_data['target'], errors='coerce').fillna(0).astype(int)
    # --- FIN DE LA CORRECCIÓN (V5) ---
    
    print(f"src.targets: Set de etiquetas creado con {len(df_target_data)} muestras (reales + negativas).")
    return df_target_data