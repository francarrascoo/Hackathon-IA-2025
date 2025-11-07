# src/rag.py
import os
from openai import OpenAI
from rank_bm25 import BM25Okapi
from src.prompts import COACH_PROMPT_TEMPLATE # Importamos el prompt que ya creamos
from dotenv import load_dotenv

# --- INICIO: Lógica de conexión de tu ejemplo ---

# 1. Configuración de GitHub Models (como en tu ejemplo)
GITHUB_ENDPOINT = "https://models.github.ai/inference"
# (Asegúrate que este es el modelo correcto para ese endpoint)
GITHUB_MODEL_NAME = "openai/gpt-4o" 

# 2. Cargar el .env (para GITHUB_TOKEN)
load_dotenv() 

try:
    # 3. Leer el token
    GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
    
    if GITHUB_TOKEN:
        print(f"src.rag: Cliente inicializado para GitHub Models ({GITHUB_MODEL_NAME}).")
        
        # 4. Inicializar el cliente (como en tu ejemplo)
        client = OpenAI(
            base_url=GITHUB_ENDPOINT,
            api_key=GITHUB_TOKEN, # Usamos el TOKEN como API Key
        )
        MODEL_TO_USE = GITHUB_MODEL_NAME
    else:
        print("src.rag: ADVERTENCIA CRÍTICA: GITHUB_TOKEN no está en .env.")
        client = None
        MODEL_TO_USE = GITHUB_MODEL_NAME

except Exception as e:
    print(f"src.rag: Error al inicializar el cliente de IA: {e}")
    client = None
    MODEL_TO_USE = GITHUB_MODEL_NAME
# --- FIN: Lógica de conexión ---


# --- INICIO: Lógica RAG (la que ya teníamos) ---

def load_knowledge_base(kb_path="kb"):
    """
    Carga todos los archivos .md de la carpeta /kb y los divide en 
    fragmentos (chunks), usando párrafos como separadores.
    """
    corpus = []
    file_sources = {}
    
    print(f"[RAG] Cargando Base de Conocimiento (KB) desde: {kb_path}")
    if not os.path.exists(kb_path):
        print(f"[RAG] Advertencia: Directorio KB '{kb_path}' no encontrado.")
        return [], {}

    for filename in os.listdir(kb_path):
        if filename.endswith(".md"):
            filepath = os.path.join(kb_path, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    chunks = content.split('\n\n')
                    
                    for i, chunk in enumerate(chunks):
                        chunk = chunk.strip()
                        if chunk:
                            chunk_id = f"{filename}_chunk_{i}"
                            corpus.append(chunk)
                            file_sources[chunk_id] = f"[Fuente: {filename}]"
            except Exception as e:
                print(f"[RAG] Error leyendo {filepath}: {e}")
                
    print(f"[RAG] KB cargada. {len(corpus)} fragmentos (chunks) indexados.")
    return corpus, file_sources

def initialize_retriever(corpus):
    """
    Inicializa el motor de búsqueda BM25 con el corpus.
    """
    if not corpus:
        print("[RAG] No hay corpus para inicializar el retriever.")
        return None
        
    tokenized_corpus = [doc.split(" ") for doc in corpus]
    retriever = BM25Okapi(tokenized_corpus)
    print("[RAG] Retriever BM25 inicializado.")
    return retriever

def search_kb(query, retriever, corpus, n_results=3):
    """
    Busca en el corpus usando el retriever BM25.
    """
    if retriever is None:
        print("[RAG] Retriever no inicializado. Devolviendo contexto vacío.")
        return "No hay información en la base de conocimiento."
        
    tokenized_query = query.split(" ")
    top_docs = retriever.get_top_n(tokenized_query, corpus, n=n_results)
    context = "\n\n---\n\n".join(top_docs)
    return context

# --- FIN: Lógica RAG ---

# --- FUNCIÓN PRINCIPAL (Modificada) ---
# (Nota: ahora es más simple, no necesita recibir 'base_url', 'api_key', 'model_name')
def get_rag_recommendation(comuna, calles_peligrosas, rag_retriever, rag_corpus):
    """
    Función principal que orquesta el RAG.
    Usa el cliente 'client' de GitHub Models inicializado globalmente.
    """
    
    if client is None:
        return {"error": "Error: El Asistente RAG no está configurado (Falta GITHUB_TOKEN)."}

    try:
        # 1. Formular la consulta de búsqueda para el RAG
        calles_str = ", ".join([c['Calle'] for c in calles_peligrosas[:2]])
        search_query = f"soluciones viales para accidentes en {calles_str} en {comuna} costos construcción impacto"

        # 2. Retrieval (Buscar en la KB)
        print(f"[RAG] Buscando contexto con query: '{search_query}'")
        contexto = search_kb(search_query, rag_retriever, rag_corpus, n_results=3)

        # 3. Augmented (Formatear el prompt)
        calles_peligrosas_str = "\n".join([
            f"- {c['Calle']} (Riesgo: {c['riesgo']}, Total Accidentes: {c['total_accidents']})" 
            for c in calles_peligrosas
        ])
        
        # Usamos el PROMPT_TEMPLATE que ya teníamos
        prompt_final = COACH_PROMPT_TEMPLATE.format(
            comuna=comuna,
            calles_peligrosas=calles_peligrosas_str,
            contexto=contexto
        )

        # 4. Generation (Llamar al LLM)
        print(f"[RAG] Llamando a {GITHUB_ENDPOINT} con modelo {MODEL_TO_USE}...")
        
        completion = client.chat.completions.create(
            model=MODEL_TO_USE, 
            messages=[
                {"role": "system", "content": f"Eres un asistente experto en seguridad vial para {comuna}. Basa tu respuesta *únicamente* en el contexto proporcionado. Cita tus fuentes."},
                {"role": "user", "content": prompt_final}
            ],
            temperature=0.2
        )
        
        plan_de_accion = completion.choices[0].message.content
        
        return {
            "plan_de_accion": plan_de_accion,
            "fuentes_consultadas": contexto
        }

    except Exception as e:
        print(f"[RAG] Error al contactar el servicio de LLM: {e}")
        return {"error": f"Error al conectar con el servicio de IA: {e}"}