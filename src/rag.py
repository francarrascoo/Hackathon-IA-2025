import os
import pandas as pd
from rank_bm25 import BM25Okapi
from src.prompts import COACH_PROMPT_TEMPLATE
from src.load import VIAL_FILE
import re
from openai import OpenAI

# --- INICIALIZAR CLIENTE OPENAI ---
try:
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    if os.environ.get("OPENAI_API_KEY") is None:
        print("src.rag: ADVERTENCIA: La variable de entorno 'OPENAI_API_KEY' no está configurada.")
        client = None
except Exception as e:
    print(f"src.rag: Error al inicializar el cliente de OpenAI: {e}")
    client = None

# --- TOKENIZADOR (para limpieza de texto) ---
def simple_tokenizer(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    return set(text.split())

# --- CARGA DE DATOS PARA EL RAG ---

def load_kb_docs(kb_path="kb/"):
    print("src.rag: Cargando Base de Conocimiento (KB) desde /kb/...")
    docs = {}
    doc_names = []
    if not os.path.exists(kb_path): return {}, None, []
    for filename in os.listdir(kb_path):
        if filename.endswith(".md"):
            with open(os.path.join(kb_path, filename), 'r', encoding='utf-8') as f:
                docs[filename] = f.read()
                doc_names.append(filename)
    if not docs: return {}, None, []
    corpus = [list(simple_tokenizer(doc)) for doc in docs.values()]
    bm25 = BM25Okapi(corpus)
    print(f"src.rag: {len(docs)} documentos .md cargados y BM25 indexado.")
    return docs, bm25, doc_names

KB_DOCS, KB_BM25, DOC_NAMES = load_kb_docs()

def load_kb_siniestros():
    print("src.rag: Cargando KB de Siniestros Reales (DESDE EL ARCHIVO ORIGINAL)...")
    try:
        df_siniestros = pd.read_csv("data/Siniestros_urbanos_biobio_2024.csv")
        COMUNAS_OBJETIVO = ["CONCEPCION", "SAN PEDRO DE LA PAZ"]
        df_siniestros = df_siniestros[df_siniestros['Comuna'].isin(COMUNAS_OBJETIVO)]
        df_siniestros['calle_simple'] = df_siniestros['Calle_Uno'].str.upper().str.strip()
        df_siniestros = df_siniestros.dropna(subset=['calle_simple'])
        siniestros_index = df_siniestros.set_index('calle_simple')
        calles_conocidas = {calle: simple_tokenizer(calle) for calle in siniestros_index.index.unique()}
        print(f"src.rag: KB de siniestros reales cargado. {len(calles_conocidas)} calles únicas indexadas.")
        return siniestros_index, calles_conocidas
    except FileNotFoundError:
        print("src.rag: ADVERTENCIA: No se pudo cargar 'Siniestros_urbanos_biobio_2024.csv' para el RAG.")
        return pd.DataFrame(columns=['FID', 'Tipo_Accid', 'Causa_Acci']), {}
    except Exception as e:
        print(f"src.rag: Error cargando KB de siniestros: {e}")
        return pd.DataFrame(columns=['FID', 'Tipo_Accid', 'Causa_Acci']), {}

KB_SINIESTROS_INDEX, CALLES_CONOCIDAS_TOKENS = load_kb_siniestros()

# --- FIN DE LA CARGA DE DATOS ---

def find_best_street_match_in_query(query_text, calles_conocidas_tokens):
    """
    Encuentra la MEJOR calle que coincida con la consulta.
    """
    if not calles_conocidas_tokens: return None
    query_tokens = simple_tokenizer(query_text)
    if not query_tokens: return None
    best_match = None
    best_score = 0
    for calle_nombre, calle_tokens in calles_conocidas_tokens.items():
        score = len(query_tokens.intersection(calle_tokens))
        if score > best_score:
            best_score = score
            best_match = calle_nombre
    if best_score > 0:
        print(f"src.rag: Mejor coincidencia específica encontrada: '{best_match}' (Score: {best_score})")
        return best_match
    return None

# --- ¡FUNCIÓN get_rag_response ACTUALIZADA! ---

def get_rag_response(query, city_context):
    """
    Proceso RAG V13: Lógica de priorización corregida.
    """
    context = ""
    sources = []
    
    # --- 1. Retrieve (Recuperar) ---
    
    # PRIMERO: Buscar si la consulta es sobre una calle específica
    calle_especifica = find_best_street_match_in_query(query, CALLES_CONOCIDAS_TOKENS)
    
    # --- CASO 1: La consulta es específica (ej. "Paicaví") ---
    if calle_especifica and calle_especifica in KB_SINIESTROS_INDEX.index:
        print(f"src.rag: Lógica RAG -> CASO 1 (Específico). Buscando: '{calle_especifica}'")
        siniestros_especificos = KB_SINIESTROS_INDEX.loc[[calle_especifica]]
        
        context += f"[Contexto de Siniestros Reales (CSV) para '{calle_especifica}']:\n"
        context += f"Se encontraron {len(siniestros_especificos)} accidentes reales en '{calle_especifica}'.\n"
        if not siniestros_especificos.empty:
            causas = siniestros_especificos['Causa_Acci'].value_counts().index[0]
            tipos = siniestros_especificos['Tipo_Accid'].value_counts().index[0]
            context += f"El tipo de accidente más común es '{tipos}'.\n"
            context += f"La causa más común es '{causas}'.\n"
        sources.append("Siniestros_urbanos_biobio_2024.csv")
    
    # --- CASO 2: La consulta es genérica (ej. "¿Qué plan sugieres?") ---
    else:
        print(f"src.rag: Lógica RAG -> CASO 2 (Genérico). Usando contexto de la ciudad.")
        hotspots = city_context.get('drivers', [])
        if hotspots:
            context += f"El análisis de la ciudad identificó estos puntos críticos (hotspots) con alta frecuencia de accidentes: {', '.join(hotspots)}."
            # (Podríamos añadir más contexto de los hotspots aquí si fuera necesario)
        else:
            context += "No se recibió un contexto de ciudad. Responda de forma general."
            
    # B. Recuperar de KB (BM25) - (Se hace siempre, como complemento)
    tokenized_query = list(simple_tokenizer(query))
    if KB_BM25 and tokenized_query:
        scores = KB_BM25.get_scores(tokenized_query)
        top_n_idx = scores.argmax()
        if scores[top_n_idx] > 0:
            doc_name = DOC_NAMES[top_n_idx]
            context += f"\n[Contexto de {doc_name}]:\n{KB_DOCS[doc_name]}\n"
            sources.append(doc_name)

    if not context:
        context = "No se encontró contexto específico en la Base de Conocimiento para esta ruta o consulta."

    # --- 2. Augment (A) ---
    prompt = COACH_PROMPT_TEMPLATE.format(context=context, query=query)
    
    # --- 3. Generate (G) ---
    print(f"--- RAG PROMPT V13 (PARA OPENAI) ---\n{prompt}\n---------------------------")
    
    if client is None:
        print("src.rag: Error: El cliente de OpenAI no está configurado.")
        return "Error: El Asistente RAG no está configurado. (Falta OPENAI_API_KEY)"

    try:
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Eres un asistente experto en seguridad vial. Tu misión es generar planes de acción para reducir accidentes, basándote *únicamente* en el contexto proporcionado. Debes citar tus fuentes. Si el contexto menciona una calle específica, enfoca tu respuesta en esa calle."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        response = completion.choices[0].message.content
        return response

    except Exception as e:
        print(f"src.rag: Error en la llamada a OpenAI: {e}")
        return f"Error al contactar al LLM: {e}"