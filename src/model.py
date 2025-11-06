import pandas as pd
import joblib
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier

# Importar nuestros propios módulos
import src.load as load
import src.targets as targets
import src.features as features
import src.eval as eval

def main():
    print("src.model: Iniciando pipeline de entrenamiento con DATOS REALES...")
    
    np.random.seed(42)
    
    # 0. Preparar datos reales si no existen
    if not os.path.exists(load.ACCIDENTES_FILE):
        print("src.model: Archivos procesados no encontrados. Ejecutando 'targets._prepare_real_data()'...")
        targets._prepare_real_data()
        print("src.model: Datos reales procesados.")

    # 1. Cargar Datos
    df_vial, df_accidentes = load.load_data()
    if df_vial is None:
        print("src.model: Abortando entrenamiento.")
        return

    # 2. Crear Targets
    df_target_data = targets.get_target_data(df_vial, df_accidentes)

    # 3. Crear Features
    df_dataset = features.create_feature_dataset(df_target_data, df_vial)
    
    X = df_dataset.drop('target', axis=1)
    y = df_dataset['target']

    # 4. Definir Pipeline
    preprocessor = features.get_preprocessor()
    model = LGBMClassifier(random_state=42, is_unbalance=True, n_estimators=150)
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    # 5. Entrenar y Validar (¡Validación Temporal REAL!)
    # Usaremos los datos de 2024.
    # Entrenar con Ene-Sep (meses 1-9), Validar con Oct-Dic (meses 10-12)
    print("src.model: Realizando validación temporal (Test en Meses 10, 11, 12)...")
    train_idx = X[X['mes'] < 10].index
    test_idx = X[X['mes'] >= 10].index

    X_train, X_test = X.loc[train_idx], X.loc[test_idx]
    y_train, y_test = y.loc[train_idx], y.loc[test_idx]
    
    if len(X_test) < 50: # Fallback si no hay suficientes datos de fin de año
        print("src.model: No hay suficientes datos para validación temporal, usando split aleatorio.")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    print(f"src.model: Entrenando en {len(X_train)} muestras, validando en {len(X_test)} muestras.")
    pipeline.fit(X_train, y_train)

    # 6. Evaluar
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    eval.print_evaluation_metrics(y_test, y_pred_proba, X_test)

    # 7. Guardar Artefacto
    os.makedirs("artifacts", exist_ok=True)
    output_path = "artifacts/concepcion_risk_model.joblib"
    joblib.dump(pipeline, output_path)
    print(f"\nsrc.model: Modelo guardado exitosamente en '{output_path}'")

if __name__ == "__main__":
    main()