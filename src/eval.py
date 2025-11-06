import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, f1_score

def print_evaluation_metrics(y_test, y_pred_proba, X_test):
    """
    Calcula e imprime las métricas de evaluación del desafío.
    """
    
    # Métricas Principales [cite: 47, 51, 52]
    auroc = roc_auc_score(y_test, y_pred_proba)
    auprc = average_precision_score(y_test, y_pred_proba)
    brier = brier_score_loss(y_test, y_pred_proba)

    print("\n--- Resultados de Evaluación del Modelo (Concepción) ---")
    print(f"  A1. Métrica AUROC: {auroc:.4f} [Puntaje: {12 if auroc > 0.8 else 7}]")
    print(f"  A2. Métrica AUPRC (Precisión-Recall): {auprc:.4f}")
    print(f"  A2. Métrica Brier Score (Calibración): {brier:.4f} [Puntaje: {6 if brier < 0.12 else 3}]")
    
    # Análisis de Fairness (Gap Absoluto) [cite: 53]
    # Comparamos el F1-score entre vías 'primary' y 'residential'
    try:
        df_eval = X_test.copy()
        df_eval['y_test'] = y_test.values
        df_eval['y_pred'] = (y_pred_proba > 0.5).astype(int)
        
        # Filtrar por grupos
        primary_mask = df_eval['type'] == 'primary'
        residential_mask = df_eval['type'] == 'residential'
        
        if primary_mask.sum() > 0 and residential_mask.sum() > 0:
            f1_primary = f1_score(df_eval[primary_mask]['y_test'], df_eval[primary_mask]['y_pred'])
            f1_residential = f1_score(df_eval[residential_mask]['y_test'], df_eval[residential_mask]['y_pred'])
            gap_absoluto = abs(f1_primary - f1_residential)
            
            print(f"--- D3. Análisis de Fairness (Gap Absoluto F1-Score) ---")
            print(f"  F1 Vías Primarias: {f1_primary:.4f}")
            print(f"  F1 Vías Residenciales: {f1_residential:.4f}")
            print(f"  Gap Absoluto: {gap_absoluto:.4f}")
        else:
            print("--- D3. Análisis de Fairness (Gap Absoluto F1-Score) ---")
            print("  No hay suficientes datos de vías 'primary' o 'residential' en el set de test para calcular el gap.")
    
    except Exception as e:
        print(f"  No se pudo calcular el Gap de Fairness: {e}")
        
    print("-----------------------------------------------------")