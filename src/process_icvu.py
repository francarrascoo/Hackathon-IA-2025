import pandas as pd
import os
import glob

# --- Configuración ---

# --- CORRECCIÓN AQUÍ ---
# Le decimos a glob que busque DENTRO de la carpeta 'data/'
ICVU_FILES = glob.glob('data/*Calidad_de_Vida_Urbana*.csv')
# --- FIN DE LA CORRECCIÓN ---

KB_OUTPUT_FILE = 'kb/datos_icvu.md'
os.makedirs('kb', exist_ok=True) # Asegurarse que la carpeta /kb exista

# Mapeo de columnas (basado en el análisis del CSV)
BASE_COLS = {
    'NOM_COMUNA': 'Comuna',
    'VIV_EN': 'Vivienda y Entorno',
    'CON_MOV': 'Conectividad y Movilidad',
    'AMB_NEG': 'Ambiente de Negocios'
}

def load_and_process_icvu(files):
    """
    Carga todos los CSV de ICVU, los estandariza y promedia por comuna.
    """
    all_dfs = []
    
    if not files:
        print("Error: No se encontraron archivos 'data/*Calidad_de_Vida_Urbana*.csv'.")
        print("Asegúrate de haber guardado los CSV del ICVU dentro de la carpeta 'data/'.")
        return None

    print(f"Archivos ICVU encontrados: {files}")

    for file in files:
        try:
            # Extraer el año del nombre del archivo (ej. '...2021.csv')
            # Usamos os.path.basename para quitar 'data/' del nombre primero
            filename_only = os.path.basename(file)
            year = filename_only.split('.')[0][-4:]
            
            if not year.isdigit():
                print(f"Omitiendo archivo con nombre no estándar: {file}")
                continue

            # Intentar leer con codificación UTF-8
            try:
                # 'file' ya incluye la ruta 'data/', así que esto funciona
                df = pd.read_csv(file) 
            except UnicodeDecodeError:
                print(f"Falló UTF-8 para {file}, intentando con 'latin1'")
                df = pd.read_csv(file, encoding='latin1')
            
            df.columns = [col.strip() for col in df.columns]
            
            current_cols_map = BASE_COLS.copy()
            icvu_col_name = f'ICVU_{year}'
            
            if icvu_col_name in df.columns:
                current_cols_map[icvu_col_name] = 'ICVU_Total'
            else:
                print(f"Advertencia: No se encontró '{icvu_col_name}' en {file}")

            df_renamed = df.rename(columns=current_cols_map)
            
            cols_to_select = [col for col in current_cols_map.values() if col in df_renamed.columns]
            df_clean = df_renamed[cols_to_select].copy()
            
            all_dfs.append(df_clean)
            
        except Exception as e:
            print(f"Error procesando el archivo {file}: {e}")

    if not all_dfs:
        print("Error: No se pudo procesar ningún archivo ICVU.")
        return None

    full_df = pd.concat(all_dfs, ignore_index=True)
    full_df['Comuna'] = full_df['Comuna'].str.upper().str.strip()
    
    df_avg = full_df.groupby('Comuna').mean(numeric_only=True).reset_index()
    
    print("Procesamiento de ICVU completado. Promedios por comuna:")
    print(df_avg.head())
    
    return df_avg

def write_kb_file(df, output_file):
    """
    Escribe el DataFrame procesado en un archivo .md para el RAG.
    """
    if df is None:
        print("No hay datos para escribir en el archivo KB.")
        return

    print(f"Escribiendo datos de ICVU en: {output_file}")
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# Ficha: Índice de Calidad de Vida Urbana (ICVU) - Promedio 2021-2023\n\n")
            f.write("Fuente: Datos del Observatorio de Ciudades UC (OCUC).\n\n")
            f.write("Esta ficha contiene los puntajes promedio (0-100) para las comunas analizadas.\n\n")
            
            for _, row in df.iterrows():
                f.write(f"## Datos ICVU para: {row['Comuna']}\n")
                f.write(f"- **Comuna:** {row['Comuna']}\n")
                
                if 'ICVU_Total' in row and pd.notna(row['ICVU_Total']):
                    f.write(f"- **ICVU Total (Promedio):** {row['ICVU_Total']:.2f}\n")
                if 'Vivienda y Entorno' in row and pd.notna(row['Vivienda y Entorno']):
                    f.write(f"- **Impacto Habitacional (Vivienda y Entorno):** {row['Vivienda y Entorno']:.2f}\n")
                if 'Conectividad y Movilidad' in row and pd.notna(row['Conectividad y Movilidad']):
                    f.write(f"- **Impacto Movilidad (Conectividad y Movilidad):** {row['Conectividad y Movilidad']:.2f}\n")
                if 'Ambiente de Negocios' in row and pd.notna(row['Ambiente de Negocios']):
                    f.write(f"- **Impacto Económico (Ambiente de Negocios):** {row['Ambiente de Negocios']:.2f}\n")
                
                f.write("\n")
                
        print(f"¡Éxito! Archivo {output_file} creado.")
        
    except Exception as e:
        print(f"Error al escribir el archivo {output_file}: {e}")

# --- Ejecución Principal ---
if __name__ == "__main__":
    processed_data = load_and_process_icvu(ICVU_FILES)
    write_kb_file(processed_data, KB_OUTPUT_FILE)