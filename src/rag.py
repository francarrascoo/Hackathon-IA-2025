import os
import re
import unicodedata  # Importamos la librería para quitar acentos
from openai import OpenAI
from rank_bm25 import BM25Okapi
from src.prompts import COACH_PROMPT_TEMPLATE
from dotenv import load_dotenv

# --- Configuración de GitHub Models ---
GITHUB_ENDPOINT = "https://models.github.ai/inference"
GITHUB_MODEL_NAME = "openai/gpt-4o" 
load_dotenv() 

try:
    GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
    if GITHUB_TOKEN:
        print(f"src.rag: Cliente inicializado para GitHub Models ({GITHUB_MODEL_NAME}).")
        client = OpenAI(
            base_url=GITHUB_ENDPOINT,
            api_key=GITHUB_TOKEN, 
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

# --- INICIO DE LA CORRECCIÓN: TOKENIZADOR CON NORMALIZACIÓN ---
def normalize_text(text):
    """
    Pasa a minúsculas, quita puntuación y quita acentos.
    ej. "CONCEPCIÓN:" -> "concepcion"
    """
    text = str(text).lower() # Pasa a minúsculas
    # Quita acentos (diacríticos)
    text = ''.join(c for c in unicodedata.normalize('NFD', text)
                   if unicodedata.category(c) != 'Mn')
    text = re.sub(r'[^\w\s]', '', text) # Quita puntuación
    return text

def simple_tokenizer(text):
    """
    Limpia el texto usando el normalizador y lo divide en palabras.
    """
    text = normalize_text(text)
    return text.split()
# --- FIN DE LA CORRECCIÓN ---


def load_knowledge_base(kb_path="kb"):
    """
    Carga todos los archivos .md de la carpeta /kb.
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
    Inicializa el motor de búsqueda BM25 con el corpus *tokenizado*.
    """
    if not corpus:
        print("[RAG] No hay corpus para inicializar el retriever.")
        return None
    
    # Usamos el tokenizador inteligente (que ahora quita acentos)
    tokenized_corpus = [simple_tokenizer(doc) for doc in corpus]
    retriever = BM25Okapi(tokenized_corpus)
    print("[RAG] Retriever BM25 inicializado con tokenizador (normalizado).")
    return retriever

def search_kb(query, retriever, corpus, n_results=5):
    """
    Busca en el corpus usando el retriever BM25.
    """
    if retriever is None:
        print("[RAG] Retriever no inicializado. Devolviendo contexto vacío.")
        return "No hay información en la base de conocimiento."
        
    # Usamos el tokenizador inteligente (que ahora quita acentos)
    tokenized_query = simple_tokenizer(query)
    
    top_docs = retriever.get_top_n(tokenized_query, corpus, n=n_results)
    
    context = "\n\n---\n\n".join(top_docs)
    return context

# --- FUNCIÓN PRINCIPAL ---
def get_rag_recommendation(comuna, calles_peligrosas, rag_retriever, rag_corpus):
    """
    Función principal que orquesta el RAG.
    """
    if client is None:
        return {"error": "Error: El Asistente RAG no está configurado (Falta GITHUB_TOKEN)."}
    try:
        calles_str = ", ".join([c['Calle'] for c in calles_peligrosas[:2]])
        
        # --- CORRECCIÓN DE LA QUERY ---
        # Nos aseguramos que la comuna también esté normalizada en la query
        normalized_comuna = normalize_text(comuna)
        search_query = f"soluciones viales para accidentes en {calles_str} en {normalized_comuna} costos construcción impacto datos icvu {normalized_comuna}"
        # --- FIN CORRECCIÓN DE LA QUERY ---

        print(f"[RAG] Buscando contexto con query: '{search_query}'")
        contexto = search_kb(search_query, rag_retriever, rag_corpus, n_results=5)

        # DEBUG: Imprimir el contexto recuperado
        print("\n" + "="*50)
        print("[RAG DEBUG] Contexto recuperado para el LLM:")
        print(contexto)
        print("="*50 + "\n")

        calles_peligrosas_str = "\n".join([
            f"- {c['Calle']} (Riesgo: {c['riesgo']}, Total Accidentes: {c['total_accidents']})" 
            for c in calles_peligrosas
        ])
        
        prompt_final = COACH_PROMPT_TEMPLATE.format(
            comuna=comuna,
            calles_peligrosas=calles_peligrosas_str,
            contexto=contexto
        )
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