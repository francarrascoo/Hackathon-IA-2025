# Optimizador de Rutas Terrestres — Hackathon IA Duoc UC 2025

Este repositorio contiene una API (FastAPI) y una aplicación web (Streamlit) para identificar puntos viales con mayor riesgo de siniestros y generar un plan de acción asistido por IA (RAG) usando una base de conocimiento local.

## Estructura principal
- `api/` — FastAPI (endpoints: `/`, `/comunas`, `/predict`, `/coach`).
- `app/` — Interfaz de usuario en Streamlit (`app/app.py`).
- `src/` — Código fuente (procesamiento, RAG, entrenamiento, etc.).
- `artifacts/` — Modelos y objetos serializados (`street_risk_model.joblib`, `label_encoder.joblib`, `model_columns.joblib`).
- `data/` — CSVs usados como datos de entrada (siniestros, índices, etc.).
- `kb/` — Base de conocimiento en Markdown para el motor RAG.

---

## Requisitos
- macOS / Linux / Windows (con herramientas equivalentes)
- Python 3.12 (recomendado — hay un `environment.yml`) o al menos Python >= 3.10
- Git (opcional)
- (Opcional) Docker

Recomendado: usar Conda con `environment.yml`. Alternativa: virtualenv + `requirements.txt`.

## 1) Preparar el entorno (Conda)

```bash
conda env create -f environment.yml
conda activate hackathon-ia-2025
```

### Alternativa (venv + pip)

```bash
python3 -m venv .venv
source .venv/bin/activate   # zsh / bash
pip install -r requirements.txt
```

## 2) Verificar artefactos y datos

Asegúrate de que `artifacts/` contiene los modelos necesarios:

- `artifacts/street_risk_model.joblib`
- `artifacts/label_encoder.joblib`
- `artifacts/model_columns.joblib`

Y que `data/` y `kb/` contienen los CSV y archivos `.md` necesarios. Si faltan archivos, la API mostrará advertencias durante el arranque.

## 3) Variables de entorno (.env)

La funcionalidad RAG (endpoint `/coach`) usa un cliente hacia GitHub Models y requiere un token. Crea un archivo `.env` en la raíz con al menos:

```env
GITHUB_TOKEN=tu_github_token_para_models
```

Notas:
- `src/rag.py` inicializa un cliente con `GITHUB_TOKEN` apuntando a `https://models.github.ai/inference`.
- Si no configuras `GITHUB_TOKEN`, la API arrancará pero `/coach` devolverá un error indicando que falta configuración.

## 4) Ejecutar localmente

Hay dos procesos: la API (FastAPI/uvicorn) y la app (Streamlit). Puedes lanzarlos por separado o usar el script `startup.sh`.

Opción A — Usar `startup.sh` (inicia API en background y Streamlit en foreground):

```bash
chmod +x startup.sh
./startup.sh
```

Opción B — Ejecutar manualmente (útil en desarrollo):

```bash
# En un terminal: Ejecutar la API
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# En otro terminal: Ejecutar la UI Streamlit
streamlit run app/app.py --server.port 8501 --server.address 0.0.0.0
```

La API estará en `http://127.0.0.1:8000` y la app en `http://127.0.0.1:8501`.

## 5) Endpoints útiles (pruebas rápidas)

- GET `/` — Ruta raíz (saludo).
- GET `/comunas` — Lista de comunas cargadas.
- POST `/predict` — Predicción de puntos peligrosos para una comuna.

Ejemplo `/predict`:

```bash
curl -s -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"comuna": "CONCEPCION"}' | jq
```

Ejemplo `/coach` (RAG):

```bash
curl -s -X POST http://127.0.0.1:8000/coach \
  -H "Content-Type: application/json" \
  -d '{"comuna": "CONCEPCION", "calles_peligrosas": [{"Calle":"Calle A","riesgo":"Alto","total_accidents":10}] }' | jq
```

## 6) Troubleshooting (problemas comunes)

- Puerto ocupado (API 8000):

```bash
lsof -nP -iTCP:8000 -sTCP:LISTEN || echo "NO_LISTENER"
# Si hay un PID en escucha, matarlo (ejemplo):
# kill -9 <PID>
```

- Errores de carga de artefactos: verifica que los archivos en `artifacts/` existen y tienen permisos de lectura.
- `/comunas` vacío: confirma que `data/` contiene los CSV `Siniestros_*.csv` y revisa los logs de arranque (uvicorn) para errores de carga.
- `/coach` falla: revisa `GITHUB_TOKEN` en `.env` y que `kb/` contenga archivos `.md`.

## 7) Docker (opcional)

```bash
# Construir imagen
docker build -t hackathon-ia-2025 .

# Ejecutar con variables de entorno desde .env
docker run -p 8000:8000 -p 8501:8501 --env-file .env hackathon-ia-2025
```

## 8) Archivos importantes

- `api/main.py` — Lógica de arranque, carga de modelos, endpoints.
- `app/app.py` — Interfaz Streamlit.
- `src/rag.py` — Lógica RAG y uso de `GITHUB_TOKEN`.
- `artifacts/` — Modelos que la API carga en startup.

## 9) Siguientes pasos sugeridos

- Añadir tests automáticos para `src/` (unit + integration) y un `Makefile` o `nox`/`tox` para automatizar ejecución.
- Añadir GitHub Action para build y tests.

Si quieres, actualizo el README con instrucciones Docker más detalladas o creo scripts (`Makefile`) para simplificar los comandos de desarrollo.

---

_Generado: README actualizado para contener todo lo necesario para ejecutar la aplicación localmente._
# Hackathon-IA-2025

Este repositorio contiene una aplicación de demostración para un Dashboard de Riesgo Vial (Región del Biobío) con API (FastAPI) y frontend (Streamlit). A continuación se describen los pasos para instalar dependencias y ejecutar la aplicación en macOS y Windows —replicando exactamente el flujo que usamos en esta sesión—.

## Requisitos previos

- Python 3.10+ instalado
- git (opcional)
- En macOS: Homebrew (para instalar `libomp` requerido por LightGBM)

---

---
title: Optimizador de Rutas (Smart Cities)
sdk: docker
app_port: 8501
---

## 1) Crear y activar un entorno virtual

macOS / Linux (zsh/bash):

```bash
# Crear el entorno virtual (una vez)
python3 -m venv venv

# Activar el entorno (cada vez que abras una nueva terminal)
source venv/bin/activate
```

Windows (PowerShell):

```powershell
# Crear el entorno virtual (una vez)
python -m venv venv

# Activar el entorno (PowerShell)
.\venv\Scripts\Activate.ps1

# (o en cmd)
.\venv\Scripts\activate.bat
```

---

## 2) Instalar dependencias

Con el entorno virtual activado, instalar los paquetes del `requirements.txt`:

```bash
pip install -r requirements.txt
```

Notas macOS: LightGBM puede requerir `libomp`. Si encuentras un error con `lib_lightgbm.dylib` o `libomp.dylib`, instala `libomp` con Homebrew y vuelve a reinstalar `lightgbm`:

```bash
# instalar libomp
brew install libomp

# (si tu shell no recoge las flags) exportarlas temporalmente
export LDFLAGS="-L/opt/homebrew/opt/libomp/lib"
export CPPFLAGS="-I/opt/homebrew/opt/libomp/include"

# reinstalar lightgbm para que enlace correctamente
pip install --force-reinstall lightgbm
```

---

## 3) Generar / entrenar el modelo (necesario antes de iniciar la API)

El proyecto contiene un script que prepara los datos y entrena el modelo, generando el artefacto necesario para la API (`artifacts/biobio_risk_model_v15.joblib`). Ejecuta:

```bash
python -m src.model
```

Deberías ver mensajes de progreso y, al final, un mensaje indicando que el modelo fue guardado en `artifacts/biobio_risk_model_v15.joblib`.

---

## 4) Configurar la clave de OpenAI (RAG/LLM)

La funcionalidad RAG necesita la variable de entorno `OPENAI_API_KEY`. Crea un archivo `.env` en la raíz del proyecto con el contenido:

```env
OPENAI_API_KEY=tu-api-key-aqui
```

Luego exporta las variables desde ese archivo antes de iniciar la API y la app, o usa un loader de `.env` en tu shell. Ejemplo (macOS / Linux):

```bash
# Exportar todas las variables del .env en la sesión actual
set -a && source .env && set +a
```

En Windows (PowerShell):

```powershell
# En PowerShell puedes usar: Get-Content .env | ForEach-Object { $name, $value = $_ -split '='; Set-Item -Path env:$name -Value $value }
```

---

## 5) Iniciar la API (FastAPI)

Con el entorno virtual activado y las variables de entorno cargadas, inicia la API:

macOS / Linux:

```bash
source venv/bin/activate
set -a && source .env && set +a
python3 api/main.py
```

Windows (PowerShell):

```powershell
.\venv\Scripts\Activate.ps1
# Cargar .env en PowerShell (ver sección anterior)
python api/main.py
```

La API arranca por defecto en `http://127.0.0.1:8000`. Puedes verificar que está funcionando con:

```bash
curl http://127.0.0.1:8000/comunas
```

Deberías recibir un JSON con la lista de comunas disponibles.

---

## 6) Iniciar la aplicación Streamlit (frontend)

En otra terminal (o pestaña), con el entorno activado y `.env` cargado:

```bash
source venv/bin/activate
set -a && source .env && set +a
streamlit run app/app.py
```

Streamlit abrirá la interfaz en `http://localhost:8501` (o te indicará la URL en la terminal). Desde la barra lateral selecciona una comuna y presiona "Analizar Comuna" para que la app haga una llamada al endpoint `/predict` de la API.

---

## Instalación rápida con `requirements.txt` (pip)

Si prefieres instalar con `pip`, ya existe un `requirements.txt` en la raíz. Desde la raíz del repositorio:

```bash
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install -r requirements.txt
```

Si la instalación de `lightgbm` falla en macOS, revisa la sección `Notas macOS` en este README o usa la instalación conda más abajo.

## Instalación recomendada (conda / conda-forge)

Para evitar problemas de compilación en macOS, la forma más fiable es usar `conda` (miniconda/Anaconda) y `conda-forge`:

```bash
# Crear y activar el entorno desde environment.yml (si lo has creado)
conda env create -f environment.yml
conda activate hackathon-ia-2025
```

O bien, si ya tienes un entorno conda activo:

```bash
conda install -c conda-forge lightgbm
python -m pip install -r requirements.txt
```

## Comprobaciones rápidas

- Verificar comunas disponibles:

```bash
curl http://127.0.0.1:8000/comunas | jq
```

- Llamada a `/predict` (todos los años):

```bash
curl -s -X POST "http://127.0.0.1:8000/predict" \
	-H "Content-Type: application/json" \
	-d '{"comuna_seleccionada":"CONCEPCION"}' | jq
```

- Llamada a `/predict` (filtrando por año):

```bash
curl -s -X POST "http://127.0.0.1:8000/predict" \
	-H "Content-Type: application/json" \
	-d '{"comuna_seleccionada":"CONCEPCION","year":2022}' | jq
```

## Notas finales

- El endpoint `/predict` ahora acepta un parámetro opcional `year` en la petición JSON; si no se envía, el API usa todos los años (2021-2024). Esto evita resultados inesperados si el frontend envía un `timestamp` por defecto.
- Para que `/coach` (RAG) funcione, instala `rank_bm25` y `openai` y configura `OPENAI_API_KEY`.

