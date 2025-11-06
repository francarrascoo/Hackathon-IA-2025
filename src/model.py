# src/model.py
import pandas as pd
import joblib
import os
import numpy as np
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier

# Importar nuestros propios módulos
import src.load as load
import src.targets as targets
import src.features as features
import src.eval as eval

def main():
    print("src.model (V15): Iniciando pipeline de VALIDACIÓN TEMPORAL (Train: 2021-23, Test: 2024)...")
    
    np.random.seed(42) # Para reproducibilidad
    
    # 0. Preparar datos reales si no existen
    if not os.path.exists(load.PROCESSED_TRAIN_FILE):
        print("src.model: Archivos procesados no encontrados. Ejecutando 'targets._prepare_real_data()'...")
        targets._prepare_real_data()
        print("src.model: Datos regionales procesados y divididos.")

    # 1. Cargar Datos (Ya divididos)
    df_train, df_test = load.load_train_test_data()
    if df_train is None or df_test is None:
        print("src.model: Abortando entrenamiento. No se pudieron cargar los datos.")
        return
    if df_train.empty or df_test.empty:
        print(f"src.model: Abortando. Datos de train ({len(df_train)}) o test ({len(df_test)}) están vacíos.")
        return

    # 2. Separar Features (X) y Target (y)
    cols_to_drop = ['target', 'FID', 'Year']
    X_train = df_train.drop(cols_to_drop, axis=1, errors='ignore')
    y_train = df_train['target']
    
    X_test = df_test.drop(cols_to_drop, axis=1, errors='ignore')
    y_test = df_test['target']

    print(f"src.model: Entrenando en {len(X_train)} muestras, validando en {len(X_test)} muestras.")

    # 3. Definir Pipeline
    preprocessor = features.get_preprocessor() # (Obtiene el preprocesador que incluye 'Comuna')
    model = LGBMClassifier(random_state=42, is_unbalance=True, n_estimators=150)
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    # 4. Entrenar
    print("src.model: Iniciando entrenamiento del modelo...")
    pipeline.fit(X_train, y_train)

    # 5. Evaluar
    print("src.model: Evaluando modelo en los datos de 2024...")
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    eval.print_evaluation_metrics(y_test, y_pred_proba, X_test)

    # 6. Guardar Artefacto
    os.makedirs("artifacts", exist_ok=True)
    output_path = "artifacts/biobio_risk_model_v15.joblib"
    joblib.dump(pipeline, output_path)
    print(f"\nsrc.model: Modelo REGIONAL (entrenado 2021-23) guardado en '{output_path}'")

if __name__ == "__main__":
    main()