# src/rag.py
import os
import pandas as pd
from rank_bm25 import BM25Okapi
from src.prompts import COACH_PROMPT_TEMPLATE
from src.load import VIAL_FILE # Solo para mapeo FID->Nombre
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

# 1. Cargar la Base de Conocimiento (KB) de archivos .md
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

# 2. Cargar la Base de Conocimiento (KB) de Siniestros Reales
def load_kb_siniestros():
    print("src.rag (V15): Cargando KB de Siniestros Reales (2021-2024)...")
    
    # Copia de la lógica de homologación de src/targets.py
    COLUMN_MAP = {
        'Fecha': ('Fecha', 'Fecha', 'Fecha', 'Fecha'),
        'Comuna': ('Comuna_1', 'Comuna', 'Comuna', 'Comuna'),
        'Calle_Uno': ('Calle_Un_1', 'Calle_Uno', 'Calle_Uno', 'Calle_Uno'),
        'Tipo_Accid': ('Tipo_Accid', 'Tipo_Accid', 'Tipo_Accid', 'Tipo_Accid'),
        'Causa_Acci': ('Causa', 'Causa', 'Causa', 'Causa_Acci') 
    }
    FILES_SINIESTROS = {
        2021: "data/Siniestros_de_tránsito,_región_del_Biobío,_Chile,_2021..csv",
        2022: "data/Siniestros_Individuales_Biobio_2022.csv",
        2023: "data/SIniestros_individuales_REGION_DEL_BIO_BIO_2023.csv",
        2024: "data/Siniestros_urbanos_biobio_2024.csv"
    }

    all_siniestros_dfs = []
    for year, file_path in FILES_SINIESTROS.items():
        try:
            df = pd.read_csv(file_path, low_memory=False, encoding='latin1')
            df_homologado = pd.DataFrame()
            map_index = year - 2021
            for standard_col, source_cols in COLUMN_MAP.items():
                col_name_to_try = source_cols[map_index]
                if col_name_to_try in df.columns:
                    df_homologado[standard_col] = df[col_name_to_try]
            all_siniestros_dfs.append(df_homologado.dropna(subset=['Comuna', 'Calle_Uno']))
        except FileNotFoundError:
            print(f"src.rag: ADVERTENCIA: No se encontró {file_path}")
    
    if not all_siniestros_dfs:
        print("src.rag: No se cargó ningún archivo de siniestros.")
        return pd.DataFrame(columns=['Comuna', 'calle_simple', 'Tipo_Accid', 'Causa_Acci']), {}

    df_siniestros_full = pd.concat(all_siniestros_dfs, ignore_index=True)
    df_siniestros_full['Comuna'] = df_siniestros_full['Comuna'].str.upper().str.strip()
    df_siniestros_full['calle_simple'] = df_siniestros_full['Calle_Uno'].str.upper().str.strip()
    df_siniestros_full = df_siniestros_full.dropna(subset=['calle_simple', 'Comuna'])
    
    siniestros_index = df_siniestros_full.set_index(['Comuna', 'calle_simple'])
    calles_conocidas = {calle: simple_tokenizer(calle) for calle in df_siniestros_full['calle_simple'].unique()}
    
    print(f"src.rag: KB de siniestros (2021-2024) cargado. {len(calles_conocidas)} calles únicas indexadas.")
    return siniestros_index, calles_conocidas

KB_SINIESTROS_INDEX, CALLES_CONOCIDAS_TOKENS = load_kb_siniestros()

# --- FIN DE LA CARGA DE DATOS ---

def find_best_street_match_in_query(query_text, calles_conocidas_tokens):
    """Encuentra la MEJOR calle que coincida con la consulta."""
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

def get_rag_response(query, city_context, comuna_seleccionada):
    """
    Proceso RAG V15: Lógica de priorización CON FILTRO DE COMUNA.
    """
    context = ""
    sources = []
    
    # --- 1. Retrieve (Recuperar) ---
    
    # PRIMERO: Filtrar la base de siniestros POR LA COMUNA SELECCIONADA
    try:
        KB_SINIESTROS_COMUNA = KB_SINIESTROS_INDEX.loc[comuna_seleccionada.upper()]
    except KeyError:
        print(f"src.rag: No hay siniestros en el índice para la comuna {comuna_seleccionada}")
        KB_SINIESTROS_COMUNA = pd.DataFrame(columns=['Tipo_Accid', 'Causa_Acci'])
    except Exception as e:
        print(f"src.rag: Error al filtrar KB por comuna: {e}")
        KB_SINIESTROS_COMUNA = pd.DataFrame(columns=['Tipo_Accid', 'Causa_Acci'])

    # SEGUNDO: Buscar si la consulta es sobre una calle específica
    calle_especifica = find_best_street_match_in_query(query, CALLES_CONOCIDAS_TOKENS)
    
    # --- CASO 1: La consulta es específica (ej. "Paicaví") ---
    if calle_especifica and not KB_SINIESTROS_COMUNA.empty and calle_especifica in KB_SINIESTROS_COMUNA.index:
        print(f"src.rag: Lógica RAG -> CASO 1 (Específico). Buscando: '{calle_especifica}' EN '{comuna_seleccionada}'")
        siniestros_especificos = KB_SINIESTROS_COMUNA.loc[[calle_especifica]]
        
        context += f"[Contexto de Siniestros Reales (CSV) para '{calle_especifica}' en {comuna_seleccionada}]:\n"
        context += f"Se encontraron {len(siniestros_especificos)} accidentes reales (2021-2024) en '{calle_especifica}'.\n"
        if not siniestros_especificos.empty:
            causas = siniestros_especificos['Causa_Acci'].value_counts().index[0]
            tipos = siniestros_especificos['Tipo_Accid'].value_counts().index[0]
            context += f"El tipo de accidente más común es '{tipos}'.\n"
            context += f"La causa más común es '{causas}'.\n"
        sources.append("Siniestros (2021-2024)")
    
    # --- CASO 2: La consulta es genérica (ej. "¿Qué plan sugieres?") ---
    else:
        print(f"src.rag: Lógica RAG -> CASO 2 (Genérico). Usando contexto de {comuna_seleccionada}.")
        hotspots = city_context.get('drivers', [])
        if hotspots:
            context += f"El análisis de la comuna '{comuna_seleccionada}' (datos 2024) identificó estos puntos críticos (hotspots): {', '.join(hotspots)}."
            
            hotspot_calle = hotspots[0]
            if not KB_SINIESTROS_COMUNA.empty and hotspot_calle in KB_SINIESTROS_COMUNA.index:
                siniestros_hotspot = KB_SINIESTROS_COMUNA.loc[[hotspot_calle]]
                causa_hotspot = siniestros_hotspot['Causa_Acci'].value_counts().index[0]
                context += f" La causa principal de accidentes en '{hotspot_calle}' (datos históricos) es: '{causa_hotspot}'."
                sources.append("Siniestros (2021-2024)")
        else:
            context += f"No se recibió un contexto de hotspots para {comuna_seleccionada}. Responda de forma general."
            
    # B. Recuperar de KB (BM25)
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
    prompt = COACH_PROMPT_TEMPLATE.format(context=context, query=query, comuna_seleccionada=comuna_seleccionada)
    
    # --- 3. Generate (G) ---
    print(f"--- RAG PROMPT V15 (PARA OPENAI) ---\n{prompt}\n---------------------------")
    
    if client is None:
        return "Error: El Asistente RAG no está configurado. (Falta OPENAI_API_KEY)"

    try:
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"Eres un asistente experto en seguridad vial. Tu misión es generar un plan de acción para la comuna de {comuna_seleccionada}. Basa tu respuesta *únicamente* en el contexto proporcionado. Cita tus fuentes (ej. [fuente: Siniestros (2021-2024)])."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        response = completion.choices[0].message.content
        return response

    except Exception as e:
        print(f"src.rag: Error en la llamada a OpenAI: {e}")
        return f"Error al contactar al LLM: {e}"