# api/main.py

import pandas as pd
import joblib
import os
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Importar las nuevas funciones de RAG
from src.rag import load_knowledge_base, initialize_retriever, get_rag_recommendation

# Cargar variables de entorno del archivo .env
load_dotenv()

# --- Modelos de Datos (Pydantic) ---
# ... (igual que antes) ...
class PredictRequest(BaseModel):
    comuna: str

class CoachRequest(BaseModel):
    comuna: str
    calles_peligrosas: list

# --- Funciones de Carga y Procesamiento ---
# ... (igual que antes) ...
DATA_FILES = [
    'Siniestros_2021.csv', 'Siniestros_2022.csv',
    'Siniestros_2023.csv', 'Siniestros_2024.csv'
]
ARTIFACTS_DIR = 'artifacts'

# En api/main.py
# ... (imports y otros) ...

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
            
            # --- NUEVO: Renombrar columnas a un formato estándar ---
            df = df.rename(columns=COLUMN_MAP)
            
            if 'fecha' in df.columns:
                df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')
                df_list.append(df)
            else:
                print(f"Advertencia: 'fecha' no encontrada en {path} (después de renombrar)")
        except FileNotFoundError:
            print(f"Error: Archivo no encontrado {path}")
        except Exception as e:
            print(f"Error cargando {path}: {e}")
            
    if not df_list:
        print("Error: No se pudieron cargar datos.")
        return pd.DataFrame()
        
    full_df = pd.concat(df_list, ignore_index=True)
    
    # --- CORRECCIÓN: Usar nombres estándar en minúsculas ---
    columnas_requeridas = ['fecha', 'comuna', 'calle_uno', 'tipo_accid']
    full_df = full_df.dropna(subset=columnas_requeridas)
    
    return full_df

def create_features_by_location(df):
    """
    Agrega los datos por ubicación (Comuna, Calle) y crea features 
    para el modelo de ML, usando nombres estándar en minúsculas.
    """
    if df.empty:
        return pd.DataFrame()

    # --- INICIO DE LA CORRECCIÓN V3 (MÁS ROBUSTA) ---
    # Convertimos la columna a string para usar .str.contains
    # Esto evita errores si hay números o nulos
    df['tipo_accid'] = df['tipo_accid'].astype(str)

    agg_ops = {
        'fecha': 'count',
        'tipo_accid': [
            # Usamos str.contains para ignorar mayúsculas/minúsculas y ser más flexibles
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
    
    return location_df.reset_index()


# --- Contexto de Vida de la Aplicación (Lifespan) ---

@asynccontextmanager
async def lifespan(app: FastAPI): 
    print("--- Evento de Inicio (Startup) ---")
    
    # 1. Cargar artefactos del modelo ML (¡La IA se carga en rag.py ahora!)
    print("Cargando artefactos del modelo ML...")
    try:
        app.state.model = joblib.load(os.path.join(ARTIFACTS_DIR, 'street_risk_model.joblib'))
        app.state.label_encoder = joblib.load(os.path.join(ARTIFACTS_DIR, 'label_encoder.joblib'))
        app.state.model_columns = joblib.load(os.path.join(ARTIFACTS_DIR, 'model_columns.joblib'))
        print("Modelo ML cargado.")
    except FileNotFoundError as e:
        print(f"Error CRÍTICO: No se encontraron artefactos del modelo. {e}")
        app.state.model = None
        
    # 2. Cargar datos de siniestros
    print("Cargando y procesando datos de siniestros...")
    raw_data = load_data(DATA_FILES)
    if not raw_data.empty:
        app.state.features_df = create_features_by_location(raw_data)
        app.state.comunas = sorted(list(app.state.features_df['comuna'].unique()))
        print(f"Datos procesados. {len(app.state.features_df)} ubicaciones cargadas.")
    else:
        print("Error CRÍTICO: No se pudieron cargar datos de siniestros.")
        app.state.features_df = pd.DataFrame()
        app.state.comunas = []

    # 3. Cargar y construir el RAG Engine
    print("Inicializando RAG Engine...")
    # (Esto ahora también inicializa el cliente de GitHub Models dentro de rag.py)
    app.state.rag_corpus, app.state.file_sources = load_knowledge_base(kb_path="kb")
    app.state.rag_retriever = initialize_retriever(app.state.rag_corpus)
    print("RAG Engine listo.")

    print("--- Inicio completado. Listo para recibir peticiones. ---")
    
    yield
    
    print("--- Evento de Apagado (Shutdown) ---")
    # Limpiar todo
    app.state.model = None
    app.state.label_encoder = None
    app.state.features_df = None
    app.state.comunas = None
    app.state.rag_corpus = None
    app.state.rag_retriever = None
    print("Recursos liberados.")

# --- Inicialización de la App ---
app = FastAPI(
    title="Optimizador de Rutas Terrestres - Hackathon DuocUC",
    description="API para predecir puntos de riesgo y generar recomendaciones.",
    version="1.0.0",
    lifespan=lifespan
)

# --- Endpoints ---

@app.get("/")
async def root():
    return {"message": "Bienvenido a la API del Optimizador de Rutas Terrestres"}

@app.get("/comunas", tags=["Predicción"])
async def get_comunas(request: Request):
    if not request.app.state.comunas:
        raise HTTPException(status_code=404, detail="No hay datos de comunas cargados.")
    return {"comunas": request.app.state.comunas}

@app.post("/predict", tags=["Predicción"])
async def predict_risk(request: PredictRequest, r: Request):
    """
    Recibe una comuna y devuelve una lista de los puntos (calles) 
    peligrosos identificados por el modelo de ML.
    """
    try:
        if r.app.state.model is None or r.app.state.features_df.empty:
            raise HTTPException(status_code=503, detail="Servidor no listo. Modelos o datos no cargados.")

        comuna = request.comuna
        
        # Filtramos por la comuna (estandarizada a minúsculas)
        comuna_df = r.app.state.features_df[
            r.app.state.features_df['comuna'].str.lower() == comuna.lower()
        ].copy()

        if comuna_df.empty:
            raise HTTPException(status_code=404, detail=f"Comuna '{comuna}' no encontrada o sin datos.")

        # Seleccionamos solo las columnas que el modelo espera
        X_predict = comuna_df[r.app.state.model_columns]
        
        print(f"--- DEBUG: Datos enviados al modelo (X_predict) ---\n{X_predict.head()}")

        # Realizamos la predicción
        predicciones_encoded = r.app.state.model.predict(X_predict)
        
        # Decodificamos las predicciones (ej. 0, 1, 2 -> 'Bajo', 'Medio', 'Alto')
        predicciones_labels = r.app.state.label_encoder.inverse_transform(predicciones_encoded)
        
        comuna_df['riesgo'] = predicciones_labels
        
        # Filtramos solo los puntos peligrosos (Alto y Medio)
        dangerous_streets = comuna_df[
            comuna_df['riesgo'].isin(['Alto', 'Medio'])
        ]
        
        # --- INICIO DE LA CORRECCIÓN ---
        # 1. Usamos 'calle_uno' (la columna real)
        output_df = dangerous_streets[['calle_uno', 'riesgo', 'total_accidents']].sort_values(
            by='total_accidents', ascending=False
        )
        
        # 2. Renombramos 'calle_uno' a 'Calle' para que se vea bien en el frontend
        output_df = output_df.rename(columns={'calle_uno': 'Calle'})
        # --- FIN DE LA CORRECCIÓN ---

        puntos_peligrosos = output_df.to_dict('records')
        
        return {
            "comuna": comuna,
            "puntos_peligrosos_identificados": puntos_peligrosos
        }

    except Exception as e:
        # Esto nos ayudará a ver el error real en el futuro
        print(f"ERROR EN /predict: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno durante la predicción: {e}")


@app.post("/coach", tags=["Coach (RAG)"])
async def get_coach_recommendation(request: CoachRequest, r: Request):
    """
    Recibe los puntos peligrosos y devuelve un plan de acción
    generado por el sistema RAG (LLM + KB Local).
    """
    
    if r.app.state.rag_retriever is None:
        raise HTTPException(status_code=503, detail="Servicio RAG no inicializado.")
    
    # --- INICIO DE LA CORRECCIÓN ---
    # Añadimos una validación para rechazar listas vacías.
    if not request.calles_peligrosas:
        print("Rechazando solicitud a /coach: la lista de calles peligrosas está vacía.")
        raise HTTPException(status_code=400, detail="No se proporcionaron calles peligrosas para el análisis.")
    # --- FIN DE LA CORRECCIÓN ---

    print(f"Iniciando RAG para comuna: {request.comuna}")
    
    rag_result = get_rag_recommendation(
        comuna=request.comuna,
        calles_peligrosas=request.calles_peligrosas,
        rag_retriever=r.app.state.rag_retriever,
        rag_corpus=r.app.state.rag_corpus
    )
    
    if "error" in rag_result:
        raise HTTPException(status_code=500, detail=rag_result["error"])

    return rag_result