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

print("api.main (V13): Iniciando API de Dashboard de Riesgo (Con RAG V13)...")

# --- Carga de Artefactos ---
try:
    PIPELINE = joblib.load("artifacts/concepcion_risk_model.joblib")
    print("api.main: Modelo de ML cargado.")
except FileNotFoundError:
    print("Error Crítico: 'artifacts/concepcion_risk_model.joblib' no encontrado.")
    sys.exit(1)

try:
    DF_VIAL = pd.read_csv(VIAL_FILE)
    DF_SINIESTROS = pd.read_csv("data/Siniestros_urbanos_biobio_2024.csv")
    COMUNAS_OBJETIVO = ["CONCEPCION", "SAN PEDRO DE LA PAZ"]
    DF_SINIESTROS = DF_SINIESTROS[DF_SINIESTROS['Comuna'].isin(COMUNAS_OBJETIVO)]
    DF_SINIESTROS['calle_simple'] = DF_SINIESTROS['Calle_Uno'].str.upper().str.strip().fillna("SIN NOMBRE")
    DF_VIAL['calle_simple'] = DF_VIAL['nombre'].str.upper().str.strip().fillna("SIN NOMBRE")
    print("api.main: Red vial y Siniestros Reales cargados.")
except FileNotFoundError as e:
    print(f"Error Crítico: No se pudo cargar un archivo de datos: {e}")
    sys.exit(1)

# --- Inicialización de FastAPI ---
app = FastAPI(
    title="Optimizador de Rutas (Concepción) - Duoc UC",
    description=f"API para el Desafío Hackathon 2025. {DISCLAIMER_TEXT}"
)

# --- Modelos de Datos Pydantic ---
class AnalysisRequest(BaseModel):
    ciudad: str = "Concepción"
    timestamp: str = "2025-11-06T18:00:00"

class CoachRequest(BaseModel):
    query: str
    contexto_ciudad: dict = {} # El RAG usará el contexto del análisis

# --- Endpoints de la API ---

@app.get("/streets")
def get_streets_list():
    """Devuelve una lista de todas las calles únicas."""
    calles = sorted(list(DF_VIAL['calle_simple'].unique()))
    return {"streets": [c for c in calles if c != "SIN NOMBRE"]}


@app.post("/predict")
def get_city_analysis(request: AnalysisRequest):
    """
    Cumple las "Funcionalidades Requeridas" 1 y 2.
    """
    print("api.main: /predict - Iniciando Estimación de Riesgo.")
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

    print("api.main: /predict - Buscando Patrones (Explicabilidad).")
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
    
    score = DF_SINIESTROS.shape[0] / DF_VIAL.shape[0] 
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
    Ahora pasa el contexto y la consulta por separado.
    """
    
    # --- ¡ESTE ES EL CAMBIO! ---
    # Ya no "contaminamos" el query. Pasamos la consulta
    # y el contexto por separado al RAG.
    response = get_rag_response(
        query=request.query, 
        city_context=request.contexto_ciudad
    )
    # --- FIN DEL CAMBIO ---

    return {"plan_textual": response}


if __name__ == "__main__":
    print("Iniciando servidor FastAPI en http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)