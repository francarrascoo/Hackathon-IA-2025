# Hackathon-IA-2025

Este repositorio contiene una aplicación de demostración para un Dashboard de Riesgo Vial (Región del Biobío) con API (FastAPI) y frontend (Streamlit). A continuación se describen los pasos para instalar dependencias y ejecutar la aplicación en macOS y Windows —replicando exactamente el flujo que usamos en esta sesión—.

## Requisitos previos

- Python 3.10+ instalado
- git (opcional)
- En macOS: Homebrew (para instalar `libomp` requerido por LightGBM)

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

## Problemas conocidos y soluciones

- Error al importar módulos después de activar venv: asegúrate de activar correctamente el entorno virtual antes de ejecutar comandos.
- Error de LightGBM relacionado con `libomp.dylib`: instala `libomp` con Homebrew y reinstala `lightgbm` (ver sección 2).
- Puertos ocupados (p.ej. 8000): mata procesos que usen el puerto antes de arrancar:

```bash
lsof -ti:8000 | xargs kill -9
```

- Si la llamada al RAG falla indicando que falta la clave de OpenAI, revisa que `OPENAI_API_KEY` esté correctamente presente en el entorno donde se ejecuta la API.

---

## Comprobación rápida (checklist)

- [ ] Crear/activar `venv`
- [ ] pip install -r requirements.txt
- [ ] Ejecutar `python -m src.model` (genera `artifacts/biobio_risk_model_v15.joblib`)
- [ ] Crear `.env` con `OPENAI_API_KEY`
- [ ] Iniciar API: `python3 api/main.py`
- [ ] Iniciar Streamlit: `streamlit run app/app.py`

---

Si prefieres, puedo añadir instrucciones adicionales (por ejemplo, cómo ejecutar en modo desarrollo con `uvicorn --reload`, o cómo generar un archivo `.env.example` sin exponer la clave). ¿Quieres que agregue eso también?
