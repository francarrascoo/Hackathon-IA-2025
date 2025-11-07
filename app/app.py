# app/app.py (Corregido)

import streamlit as st
import requests
import pandas as pd
import json

st.set_page_config(
    page_title="Optimizador de Rutas",
    page_icon="🚨",
    layout="wide",
)

API_URL = "http://127.0.0.1:8000"

# Inicialización del Estado de Sesión
if 'comunas' not in st.session_state:
    st.session_state.comunas = []
if 'selected_comuna' not in st.session_state:
    st.session_state.selected_comuna = None
if 'hotspots_df' not in st.session_state:
    st.session_state.hotspots_df = None
if 'coach_plan' not in st.session_state:
    st.session_state.coach_plan = None
if 'fuentes' not in st.session_state:
    st.session_state.fuentes = None

@st.cache_data(ttl=3600)
def get_comunas():
    try:
        response = requests.get(f"{API_URL}/comunas")
        if response.status_code == 200:
            return response.json().get("comunas", [])
        else:
            st.error(f"Error al cargar comunas: {response.status_code}")
            return []
    except requests.ConnectionError:
        st.error(
            "Error de Conexión: No se pudo conectar a la API. "
            "¿Ejecutaste 'uvicorn api.main:app --reload'?"
        )
        return []

def run_prediction(comuna):
    try:
        response = requests.post(f"{API_URL}/predict", json={"comuna": comuna})
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error en la predicción: {response.status_code} - {response.text}")
            return None
    except requests.ConnectionError:
        st.error("Error de Conexión: No se pudo conectar a la API de predicción.")
        return None

def get_coach_plan(comuna, hotspots_list):
    try:
        payload = {
            "comuna": comuna,
            "calles_peligrosas": hotspots_list
        }
        response = requests.post(f"{API_URL}/coach", json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error en el coach: {response.status_code} - {response.text}")
            return None
    except requests.ConnectionError:
        st.error("Error de Conexión: No se pudo conectar a la API del coach.")
        return None

# --- Interfaz de Usuario (UI) ---
st.title("🚨 Optimizador de Rutas y Prevención de Accidentes")
st.caption("Hackathon IA Duoc UC 2025 - Smart Cities")

if not st.session_state.comunas:
    st.session_state.comunas = get_comunas()

col1, col2 = st.columns([1, 2])

with col1:
    st.header("1. Análisis de Riesgo")
    st.markdown(
        "Selecciona una comuna para identificar los puntos críticos (calles) "
        "con mayor riesgo de accidentes."
    )
    
    if not st.session_state.comunas:
        st.warning("No se pudieron cargar las comunas desde la API.")
    else:
        selected_comuna = st.selectbox(
            "Selecciona una Comuna",
            options=st.session_state.comunas
        )

        analysis_button = st.button("Analizar Riesgo", type="primary", use_container_width=True)

        if analysis_button and selected_comuna:
            with st.spinner(f"Analizando {selected_comuna}..."):
                st.session_state.hotspots_df = None
                st.session_state.coach_plan = None
                st.session_state.fuentes = None

                prediction_result = run_prediction(selected_comuna)
                
                if prediction_result:
                    hotspots_data = prediction_result.get("puntos_peligrosos_identificados")
                    st.session_state.selected_comuna = selected_comuna
                    
                    if hotspots_data: # Si la lista NO está vacía
                        st.session_state.hotspots_df = pd.DataFrame(hotspots_data)
                    else:
                        # Si está vacía, creamos un DF vacío para mostrar el mensaje
                        st.session_state.hotspots_df = pd.DataFrame(columns=['Calle', 'riesgo', 'total_accidents'])

    # Mostrar el plan de acción (si ya existe)
    if st.session_state.coach_plan:
        st.divider()
        st.header("3. Plan de Acción Sugerido")
        with st.expander("Ver Plan de Acción detallado", expanded=True):
            st.markdown(st.session_state.coach_plan)
            st.caption(f"Fuentes consultadas: {st.session_state.fuentes}")
        
        st.download_button(
            "Descargar Reporte PDF",
            data="Esto sería un PDF",
            file_name=f"reporte_{st.session_state.selected_comuna}.pdf",
            mime="application/pdf",
            use_container_width=True
        )

with col2:
    st.header("2. Puntos Críticos Identificados")
    
    if st.session_state.hotspots_df is None:
        st.info("Esperando análisis de riesgo...")
    
    # --- INICIO DE LA CORRECCIÓN ---
    # Verificamos si el DataFrame NO está vacío
    elif not st.session_state.hotspots_df.empty:
        st.markdown(f"Resultados para **{st.session_state.selected_comuna}**:")
        
        st.dataframe(
            st.session_state.hotspots_df.style.apply(
                lambda row: ["background-color: #FFC7CE"] * len(row) if row.riesgo == "Alto" 
                else (["background-color: #FFE5CC"] * len(row) if row.riesgo == "Medio" 
                else [""] * len(row)),
                axis=1
            ),
            use_container_width=True
        )
        
        st.info("El modelo prioriza calles con riesgo 'Alto' y 'Medio'.")
        st.divider()
        
        st.markdown(
            "Usa el asistente de IA para generar un plan de acción basado en "
            "la base de conocimiento local (RAG)."
        )
        coach_button = st.button("Obtener Plan de Acción (IA)", use_container_width=True)

        if coach_button:
            with st.spinner("Generando plan de acción con IA..."):
                hotspots_list = st.session_state.hotspots_df.to_dict('records')
                coach_result = get_coach_plan(
                    st.session_state.selected_comuna, 
                    hotspots_list
                )
                
                if coach_result:
                    st.session_state.coach_plan = coach_result.get("plan_de_accion")
                    st.session_state.fuentes = coach_result.get("fuentes_consultadas")
                    st.rerun() 
    
    # Si el DataFrame SÍ está vacío (significa que no se encontraron hotspots)
    else:
        st.success(f"¡Buenas noticias! No se identificaron puntos de riesgo 'Alto' o 'Medio' en {st.session_state.selected_comuna} según los umbrales actuales.")
    # --- FIN DE LA CORRECCIÓN ---