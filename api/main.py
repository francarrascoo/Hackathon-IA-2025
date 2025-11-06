import pandas as pd
import joblib
import networkx as nx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
import numpy as np
import random
import uvicorn
import os
import sys

# Añadir 'src' al path para poder importar nuestros módulos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.load import VIAL_FILE, ACCIDENTES_FILE
from src.rag import get_rag_response
from src.prompts import DISCLAIMER_TEXT

print("api.main (V14): Iniciando API de Dashboard de Riesgo (Dinámico por Comuna)...")

# --- Carga de Artefactos ---
try:
    PIPELINE = joblib.load("artifacts/concepcion_risk_model.joblib")
    print("api.main: Modelo de ML (entrenado en CCP/SPP) cargado.")
except FileNotFoundError:
    print("Error Crítico: 'artifacts/concepcion_risk_model.joblib' no encontrado.")
    sys.exit(1)

# --- Carga de Datos Completa (Toda la Región) ---
try:
    DF_VIAL_FULL = pd.read_csv("data/Red_Vial_de_Chile.csv")
    DF_SINIESTROS_FULL = pd.read_csv("data/Siniestros_urbanos_biobio_2024.csv")
    
    # Limpieza general
    DF_SINIESTROS_FULL['calle_simple'] = DF_SINIESTROS_FULL['Calle_Uno'].str.upper().str.strip().fillna("SIN NOMBRE")
    DF_VIAL_FULL['calle_simple'] = DF_VIAL_FULL['nombre'].str.upper().str.strip().fillna("SIN NOMBRE")
    
    # Extraer lista de comunas únicas de la Región del Biobío
    COMUNAS_DISPONIBLES = sorted(list(DF_SINIESTROS_FULL['Comuna'].unique()))
    
    print(f"api.main: Red vial y {len(DF_SINIESTROS_FULL)} Siniestros Reales cargados.")
    print(f"api.main: Comunas disponibles: {COMUNAS_DISPONIBLES}")

except FileNotFoundError as e:
    print(f"Error Crítico: No se pudo cargar un archivo de datos: {e}")
    sys.exit(1)

# --- Inicialización de FastAPI ---
app = FastAPI(
    title="Optimizador de Rutas (Biobío) - Duoc UC",
    description=f"API para el Desafío Hackathon 2025. {DISCLAIMER_TEXT}"
)

# --- Modelos de Datos Pydantic ---
class AnalysisRequest(BaseModel):
    comuna_seleccionada: str # Ahora la comuna es un parámetro
    timestamp: str = "2025-11-06T18:00:00"

class CoachRequest(BaseModel):
    query: str
    contexto_ciudad: dict = {}
    comuna_seleccionada: str # El RAG también necesita saber la comuna

# --- Endpoints de la API ---

@app.get("/comunas")
def get_comunas_list():
    """
    NUEVO ENDPOINT: Devuelve la lista de comunas disponibles
    en el CSV de siniestros.
    """
    return {"comunas": COMUNAS_DISPONIBLES}


@app.post("/predict")
def get_city_analysis(request: AnalysisRequest):
    """
    ENDPOINT /predict MODIFICADO
    Ahora filtra dinámicamente por la comuna seleccionada.
    """
    
    comuna = request.comuna_seleccionada
    print(f"api.main: /predict - Iniciando Análisis para: {comuna}")

    # --- FILTRADO DINÁMICO ---
    DF_SINIESTROS = DF_SINIESTROS_FULL[DF_SINIESTROS_FULL['Comuna'] == comuna].copy()
    DF_VIAL = DF_VIAL_FULL[DF_VIAL_FULL['nombre_com'] == comuna].copy()
    # --- FIN FILTRADO ---
    
    if DF_SINIESTROS.empty:
        raise HTTPException(status_code=404, detail=f"No se encontraron datos de siniestros para la comuna: {comuna}")

    # 1. ESTIMACIÓN DE RIESGO (Hotspots y Categorización)
    hotspots = DF_SINIESTROS['calle_simple'].value_counts()
    
    def categorize(count):
        if count > 20: return "Muy Frecuente"
        if count > 5: return "Común"
        return "Esporádico"

    df_hotspots = pd.DataFrame({
        'Calle': hotspots.index,
        'Nº Accidentes': hotspots.values,
        'Ocurrencia': [categorize(c) for c in hotspots.values]
    })
    
    estimacion_riesgo = df_hotspots.head(10).to_dict('records')

    # 2. EXPLICABILIDAD (Patrones)
    patron_causas = DF_SINIESTROS['Causa_Acci'].value_counts().head(3).to_dict()
    patron_tipo = DF_SINIESTROS['Tipo_Accid'].value_counts().head(3).to_dict()
    df_merged = pd.merge(
        DF_SINIESTROS,
        DF_VIAL.drop_duplicates('calle_simple'),
        on='calle_simple',
        how='left'
    )
    patron_via = df_merged['type'].value_counts().head(3).to_dict()
    
    explicabilidad = {
        "patron_causas_principales": patron_causas,
        "patron_tipo_accidente": patron_tipo,
        "patron_tipo_via": patron_via
    }
    
    score = DF_SINIESTROS.shape[0] / max(1, DF_VIAL.shape[0]) # Riesgo de la comuna
    drivers = list(hotspots.head(3).index) # Top 3 calles
    
    return {
        "score": score,
        "drivers": drivers,
        "estimacion_riesgo": estimacion_riesgo,
        "explicabilidad": explicabilidad
    }

# --- Endpoint 2: /coach (CORREGIDO) ---
@app.post("/coach")
def coach_rag_assistant(request: CoachRequest):
    """
    Cumple la "Funcionalidad Requerida" 3: Plan de Acción (RAG).
    Ahora pasa el contexto Y la comuna al RAG.
    """
    response = get_rag_response(
        query=request.query, 
        city_context=request.contexto_ciudad,
        comuna_seleccionada=request.comuna_seleccionada
    )
    return {"plan_textual": response}


if __name__ == "__main__":
    print("Iniciando servidor FastAPI en http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)