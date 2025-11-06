import pandas as pd
import numpy as np
import random
import os
from src.load import PROCESSED_TRAIN_FILE, PROCESSED_TEST_FILE, VIAL_FILE

# --- CONSTANTES DE FILTRADO (V15) ---
REGION_SINIESTROS = "REGION BIO BIO"
REGION_VIAL = "REGIÓN DEL BIOBÍO"
CSV_VIAL_NACIONAL = "data/Red_Vial_de_Chile.csv"

# Los 4 archivos de siniestros
FILES_SINIESTROS = {
    2021: "data/Siniestros_2021.csv",
    2022: "data/Siniestros_2022.csv",
    2023: "data/Siniestros_2023.csv",
    2024: "data/Siniestros_2024.csv"
}

# --- MAPA DE HOMOLOGACIÓN ---
COLUMN_MAP = {
    'Fecha': ('Fecha', 'Fecha', 'Fecha', 'Fecha'),
    'Comuna': ('Comuna_1', 'Comuna', 'Comuna', 'Comuna'),
    'Calle_Uno': ('Calle_Un_1', 'Calle_Uno', 'Calle_Uno', 'Calle_Uno'),
    'Tipo_Accid': ('Tipo_Accid', 'Tipo_Accid', 'Tipo_Accid', 'Tipo_Accid'),
    'Causa_Acci': ('Causa', 'Causa', 'Causa', 'Causa_Acci') 
}

def _load_and_homologate(year):
    """Carga un CSV de siniestro y estandariza sus columnas."""
    file_path = FILES_SINIESTROS[year]
    try:
        # Usar 'latin1' (o 'ISO-8859-1') es común para archivos con tildes
        df = pd.read_csv(file_path, low_memory=False, encoding='latin1')
        
        df_homologado = pd.DataFrame()
        map_index = year - 2021 # 0=2021, 1=2022, 2=2023, 3=2024
        
        for standard_col, source_cols in COLUMN_MAP.items():
            col_name_to_try = source_cols[map_index]
            if col_name_to_try in df.columns:
                df_homologado[standard_col] = df[col_name_to_try]
        
        df_homologado['Year'] = year
        return df_homologado.dropna(subset=['Fecha', 'Comuna', 'Calle_Uno'])
        
    except FileNotFoundError:
        print(f"src.targets: ADVERTENCIA: No se encontró el archivo {file_path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"src.targets: Error al leer {file_path}: {e}")
        return pd.DataFrame()


def _prepare_real_data():
    """
    Función de ETL: Carga los 4 CSVs, los homologa, los une a la red vial
    y los separa en train (2021-2023) y test (2024).
    """
    print(f"src.targets (V16): Iniciando ETL completo (2021-2024)...")
    
    # 1. Cargar y filtrar Red Vial
    try:
        # --- ¡AQUÍ ESTÁ LA CORRECCIÓN (V16)! ---
        # Usar 'utf-8-sig' para manejar el BOM (Byte Order Mark) que
        # a veces corrompe el nombre de la primera columna ("FID").
        df_vial_full = pd.read_csv(CSV_VIAL_NACIONAL, encoding='utf-8-sig') 
        # --- FIN DE LA CORRECCIÓN ---
        
    except FileNotFoundError:
        print(f"Error: '{CSV_VIAL_NACIONAL}' no encontrado.")
        return
    except Exception as e:
        print(f"Error al leer {CSV_VIAL_NACIONAL}: {e}")
        return

    # Debugging: Imprimir columnas para verificar
    print(f"Columnas de Red Vial: {df_vial_full.columns.tolist()}")

    # Asegurarse de que FID exista antes de continuar
    if 'FID' not in df_vial_full.columns:
        print("Error Crítico: La columna 'FID' no se encontró en 'Red_Vial_de_Chile.csv'.")
        print("Verifique el archivo o el encoding.")
        return

    df_vial_region = df_vial_full[df_vial_full['nombre_reg'] == REGION_VIAL].copy()
    df_vial_region['calle_join_key'] = df_vial_region['nombre'].str.upper().str.strip()
    df_vial_region['nombre_com'] = df_vial_region['nombre_com'].str.upper().str.strip()
    df_vial_region.to_csv(VIAL_FILE, index=False)
    print(f"src.targets: Red vial regional guardada en '{VIAL_FILE}'.")
    
    # 2. Cargar y Homologar Siniestros (2021-2024)
    all_siniestros = [_load_and_homologate(year) for year in FILES_SINIESTROS]
    df_siniestros_full = pd.concat(all_siniestros, ignore_index=True)
    df_siniestros_full['calle_join_key'] = df_siniestros_full['Calle_Uno'].str.upper().str.strip()
    df_siniestros_full['Comuna'] = df_siniestros_full['Comuna'].str.upper().str.strip()
    print(f"src.targets: {len(df_siniestros_full)} siniestros totales homologados.")

    # 3. Cruzar Siniestros con Red Vial (Join por nombre de calle Y comuna)
    df_accidentes_full = pd.merge(
        df_siniestros_full,
        df_vial_region[['FID', 'calle_join_key', 'nombre_com', 'type', 'Shape__Length']], # Esta es la línea 88
        how="inner",
        left_on=['calle_join_key', 'Comuna'],
        right_on=['calle_join_key', 'nombre_com']
    )
    print(f"src.targets: {len(df_accidentes_full)} accidentes mapeados a un FID vial.")
    
    # Corregir bug de renombrado de pandas
    if "FID_y" in df_accidentes_full.columns:
        df_accidentes_full = df_accidentes_full.rename(columns={"FID_y": "FID", "FID_x": "FID_siniestro"})
    
    # 4. Crear Etiquetas (Target=1)
    df_acc_pos = df_accidentes_full.copy()
    df_acc_pos['timestamp'] = pd.to_datetime(df_acc_pos['Fecha'], errors='coerce')
    df_acc_pos = df_acc_pos.dropna(subset=['timestamp'])
    df_acc_pos['hora_del_dia'] = df_acc_pos['timestamp'].dt.hour
    df_acc_pos['dia_semana'] = df_acc_pos['timestamp'].dt.dayofweek
    df_acc_pos['mes'] = df_acc_pos['timestamp'].dt.month
    df_acc_pos['target'] = 1
    
    cols_positivas = ['FID', 'Comuna', 'type', 'Shape__Length', 'Year', 
                      'hora_del_dia', 'dia_semana', 'mes', 'target']
    df_acc_pos = df_acc_pos[cols_positivas]

    # 5. Crear Etiquetas Negativas (Target=0)
    print("src.targets: Creando muestras negativas...")
    num_neg_samples = len(df_acc_pos) # 1:1 ratio
    neg_samples = []
    
    vial_pool = df_vial_region[['FID', 'nombre_com', 'type', 'Shape__Length']].dropna().to_dict('records')
    if not vial_pool:
        print("Error: El pool de vías para muestreo negativo está vacío.")
        return
        
    for _ in range(num_neg_samples):
        random_via = random.choice(vial_pool)
        random_year = random.choice([2021, 2022, 2023, 2024])
        neg_samples.append({
            'FID': random_via['FID'],
            'Comuna': random_via['nombre_com'],
            'type': random_via['type'],
            'Shape__Length': random_via['Shape__Length'],
            'Year': random_year,
            'hora_del_dia': random.randint(0, 23),
            'dia_semana': random.randint(0, 6),
            'mes': random.randint(1, 12),
            'target': 0
        })
    df_neg = pd.DataFrame(neg_samples)
    
    # 6. Dataset Final y Split
    df_dataset_full = pd.concat([df_acc_pos, df_neg], ignore_index=True)
    df_dataset_full['target'] = pd.to_numeric(df_dataset_full['target'], errors='coerce').fillna(0).astype(int)
    
    # --- LA VALIDACIÓN TEMPORAL ---
    df_train = df_dataset_full[df_dataset_full['Year'] < 2024]
    df_test = df_dataset_full[df_dataset_full['Year'] == 2024]
    
    # 7. Guardar
    df_train.to_csv(PROCESSED_TRAIN_FILE, index=False)
    df_test.to_csv(PROCESSED_TEST_FILE, index=False)
    
    print("-----------------------------------------------------")
    print("src.targets: ¡ETL y Validación Temporal Completados!")
    print(f"Datos de Entrenamiento: {len(df_train)} (2021-2023)")
    print(f"Datos de Prueba (Test): {len(df_test)} (2024)")
    print(f"Archivos guardados en '{PROCESSED_TRAIN_FILE}' y '{PROCESSED_TEST_FILE}'.")
    print("-----------------------------------------------------")