import streamlit as st
import requests
import pandas as pd
from datetime import datetime
import sys, os

# Añadir 'src' al path para poder importar prompts
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.prompts import DISCLAIMER_TEXT

# URL de la API (debe estar corriendo localmente)
API_URL = "http://127.0.0.1:8000"

st.set_page_config(layout="wide", page_title="Dashboard de Riesgo (Concepción)")
st.title("🛰️ Dashboard de Riesgo Vial - Hackathon Duoc UC 2025")
st.caption("App demo implementada en Streamlit" )

# --- Inicializar estado de sesión ---
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

# --- Sidebar (Configuración) ---
st.sidebar.header("Configuración de Análisis")

# 1. SELECCIÓN DE CIUDAD
# (Hardcodeado a Concepción, ya que nuestros datos son de ahí)
ciudad_seleccionada = st.sidebar.selectbox(
    "Selecciona una Ciudad",
    options=["Concepción / San Pedro de la Paz"],
    index=0
)

# 2. SELECCIÓN DE FECHA
default_time = datetime(2025, 11, 6, 18, 0) # Hora del Hackathon
time_input = st.sidebar.time_input("Hora de Análisis", value=default_time.time())
date_input = st.sidebar.date_input("Fecha de Análisis", value=default_time.date())
timestamp = f"{date_input}T{time_input}"

st.sidebar.info(f"Analizando: {ciudad_seleccionada} @ {timestamp}")
st.sidebar.markdown(DISCLAIMER_TEXT, unsafe_allow_html=True) # Disclaimer visible

# --- Layout Principal ---

# Botón para ejecutar el análisis
if st.sidebar.button("Analizar Ciudad", type="primary"):
    payload = {
        "ciudad": ciudad_seleccionada,
        "timestamp": timestamp
    }
    try:
        with st.spinner(f"Analizando {ciudad_seleccionada}... (Llamando a API: /predict)"):
            response = requests.post(f"{API_URL}/predict", json=payload)
            response.raise_for_status()
        
        st.session_state.analysis_results = response.json()
        st.session_state.messages = [] # Limpiar chat

    except requests.exceptions.RequestException as e:
        st.error(f"Error al contactar la API: {e}")
    except Exception as e:
        st.error(f"Error inesperado: {e}")

# --- Mostrar resultados del análisis (si existen) ---
if st.session_state.analysis_results:
    results = st.session_state.analysis_results
    
    # Dividir la página en 3 secciones, como pediste
    
    # --- 1. ESTIMACIÓN DE RIESGO ---
    st.header("1. Estimación de Riesgo")
    st.markdown("Identificación de puntos comunes de accidentes (Top 10 Calles):")
    
    if "estimacion_riesgo" in results:
        df_hotspots = pd.DataFrame(results['estimacion_riesgo'])
        st.dataframe(df_hotspots, use_container_width=True)
    
    # --- 2. EXPLICABILIDAD ---
    st.header("2. Explicabilidad")
    st.markdown("Búsqueda de patrones en los datos de accidentes reales:")
    
    if "explicabilidad" in results:
        exp = results['explicabilidad']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Causas Principales")
            st.write(exp.get("patron_causas_principales", {}))
        with col2:
            st.subheader("Tipos de Accidente")
            st.write(exp.get("patron_tipo_accidente", {}))
        with col3:
            st.subheader("Tipos de Vía Afectadas")
            st.write(exp.get("patron_tipo_via", {}))
    
    st.divider()

    # --- 3. PLAN DE ACCIÓN (RAG) ---
    st.header("3. Plan de Acción (Coach RAG)")
    st.markdown("Genera un plan de acción basado en los hallazgos. (Llama a API: /coach)")

    # Mostrar historial del chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input del chat
    if prompt := st.chat_input("¿Qué plan de acción sugieres basado en estos patrones?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Preparar payload para el RAG
        # (Pasamos los drivers (hotspots) al RAG como contexto)
        rag_payload = {
            "query": prompt,
            "contexto_ciudad": {
                "drivers": results.get("drivers", [])
            }
        }
        
        try:
            with st.spinner("El coach RAG está generando un plan..."):
                rag_response = requests.post(f"{API_URL}/coach", json=rag_payload)
                rag_response.raise_for_status()
                
                response_data = rag_response.json()
                bot_response = response_data.get("plan_textual", "No pude procesar la respuesta.")
            
            st.session_state.messages.append({"role": "assistant", "content": bot_response})
            with st.chat_message("assistant"):
                st.markdown(bot_response)
                
        except requests.exceptions.RequestException as e:
            st.error(f"Error al contactar la API del Coach: {e}")

else:
    st.info("Haz clic en **Analizar Ciudad** en la barra lateral para comenzar.")