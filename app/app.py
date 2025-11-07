import streamlit as st
import requests
import pandas as pd
import json

# --- Configuración de la Página y API ---
st.set_page_config(
    page_title="Optimizador de Rutas",
    page_icon="🚨",
    layout="wide",
)

# URL del backend (API)
API_URL = "http://127.0.0.1:8000"

# --- INICIALIZACIÓN DEL ESTADO DE SESIÓN ---
# Esto es clave para "recordar" los resultados entre clics de botón
if 'comunas' not in st.session_state:
    st.session_state.comunas = []
if 'selected_comuna' not in st.session_state:
    st.session_state.selected_comuna = None
if 'hotspots_df' not in st.session_state:
    st.session_state.hotspots_df = None # Aquí guardaremos el DataFrame
if 'coach_plan' not in st.session_state:
    st.session_state.coach_plan = None
if 'fuentes' not in st.session_state:
    st.session_state.fuentes = None

# --- Funciones de la App ---

@st.cache_data(ttl=3600) # Cachear la lista de comunas por 1 hora
def get_comunas():
    """Obtiene la lista de comunas desde la API."""
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
    """Ejecuta el análisis de predicción en la API."""
    try:
        # CORRECCIÓN DE FORMATO JSON
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
    """Obtiene el plan de acción desde el coach RAG."""
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

# Cargar comunas solo una vez
if not st.session_state.comunas:
    st.session_state.comunas = get_comunas()

# --- COLUMNA 1: Selección y Análisis ---
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

        # --- LÓGICA DEL BOTÓN DE ANÁLISIS ---
        if analysis_button and selected_comuna:
            with st.spinner(f"Analizando {selected_comuna}..."):
                # Limpiar resultados anteriores
                st.session_state.hotspots_df = None
                st.session_state.coach_plan = None
                st.session_state.fuentes = None

                # Llamar a la API
                prediction_result = run_prediction(selected_comuna)
                
                if prediction_result:
                    hotspots_data = prediction_result.get("puntos_peligrosos_identificados")
                    
                    if hotspots_data:
                        # Guardar DataFrame en el ESTADO DE SESIÓN
                        st.session_state.hotspots_df = pd.DataFrame(hotspots_data)
                        st.session_state.selected_comuna = selected_comuna # Guardar comuna
                    else:
                        st.info(
                            f"No se encontraron puntos de riesgo 'Alto' o 'Medio' "
                            f"en {selected_comuna}. ¡Buenas noticias!"
                        )

    # --- Mostrar el plan de acción (si ya existe) ---
    if st.session_state.coach_plan:
        st.divider()
        st.header("3. Plan de Acción Sugerido")
        with st.expander("Ver Plan de Acción detallado", expanded=True):
            st.markdown(st.session_state.coach_plan)
            st.caption(f"Fuentes consultadas: {st.session_state.fuentes}")
        
        # Botón de descarga (simulado, requiere lógica PDF)
        st.download_button(
            "Descargar Reporte PDF",
            data="Esto sería un PDF",
            file_name=f"reporte_{st.session_state.selected_comuna}.pdf",
            mime="application/pdf",
            use_container_width=True
        )

# --- COLUMNA 2: Resultados y Coach ---
with col2:
    st.header("2. Puntos Críticos Identificados")
    
    # Mostrar el DataFrame si existe en el estado de sesión
    if st.session_state.hotspots_df is not None:
        st.markdown(f"Resultados para **{st.session_state.selected_comuna}**:")
        
        # Aplicar estilo al DataFrame
        st.dataframe(
    st.session_state.hotspots_df.style.apply(
        lambda row: ["background-color: #FFC7CE"] * len(row) if row.riesgo == "Alto" 
        else (["background-color: #FFE5CC"] * len(row) if row.riesgo == "Medio" # <-- CORREGIDA
        else [""] * len(row)),
        axis=1
    ),
    use_container_width=True
)
        
        st.info("El modelo prioriza calles con riesgo 'Alto' y 'Medio'.")
        
        st.divider()
        
        # --- LÓGICA DEL BOTÓN DEL COACH ---
        st.markdown(
            "Usa el asistente de IA para generar un plan de acción basado en "
            "la base de conocimiento local (RAG)."
        )
        coach_button = st.button("Obtener Plan de Acción (IA)", use_container_width=True)

        if coach_button:
            # Validar que tengamos datos para enviar
            if st.session_state.hotspots_df is None:
                st.error(
                    "Error: No hay puntos de riesgo analizados. "
                    "Por favor, presiona 'Analizar Riesgo' primero."
                )
            else:
                with st.spinner("Generando plan de acción con IA..."):
                    # Convertir el DF a la lista que espera la API
                    hotspots_list = st.session_state.hotspots_df.to_dict('records')
                    
                    # Llamar a la API del coach
                    coach_result = get_coach_plan(
                        st.session_state.selected_comuna, 
                        hotspots_list
                    )
                    
                    if coach_result:
                        # Guardar resultados en el ESTADO DE SESIÓN
                        st.session_state.coach_plan = coach_result.get("plan_de_accion")
                        st.session_state.fuentes = coach_result.get("fuentes_consultadas")
                        
                        # Forzar un 'rerun' de Streamlit para que muestre 
                        # el plan en la columna 1
                        st.rerun() 
                        
    else:
        st.info("Esperando análisis de riesgo...")