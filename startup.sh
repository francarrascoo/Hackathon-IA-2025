#!/bin/bash

# 1. Inicia la API (uvicorn) en segundo plano
# Usamos --host 0.0.0.0 para que sea accesible dentro de Docker
# El '&' al final es crucial para que se ejecute en segundo plano
echo "Iniciando API FastAPI en puerto 8000..."
uvicorn api.main:app --host 0.0.0.0 --port 8000 &

# 2. Inicia la App (streamlit) en primer plano
# Esto mantiene el contenedor corriendo
echo "Iniciando App Streamlit en puerto 8501..."
streamlit run app/app.py --server.port 8501 --server.address 0.0.0.0