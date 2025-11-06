# src/eval.py
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, f1_score

def print_evaluation_metrics(y_test, y_pred_proba, X_test):
    """
    Calcula e imprime las métricas de evaluación del desafío.
    """
    
    # Métricas Principales
    auroc = roc_auc_score(y_test, y_pred_proba)
    auprc = average_precision_score(y_test, y_pred_proba)
    brier = brier_score_loss(y_test, y_pred_proba)

    print("\n--- Resultados de Evaluación del Modelo (Validación 2024) ---")
    print(f"  A1. Métrica AUROC: {auroc:.4f}")
    print(f"  A2. Métrica AUPRC (Precisión-Recall): {auprc:.4f}")
    print(f"  A2. Métrica Brier Score (Calibración): {brier:.4f}")
    
    # Análisis de Fairness (Gap Absoluto)
    try:
        df_eval = X_test.copy()
        df_eval['y_test'] = y_test.values
        df_eval['y_pred'] = (y_pred_proba > 0.5).astype(int)
        
        # Comparar las dos comunas principales
        concepcion_mask = df_eval['Comuna'] == 'CONCEPCION'
        spp_mask = df_eval['Comuna'] == 'SAN PEDRO DE LA PAZ'
        
        if concepcion_mask.sum() > 0 and spp_mask.sum() > 0:
            f1_concepcion = f1_score(df_eval[concepcion_mask]['y_test'], df_eval[concepcion_mask]['y_pred'])
            f1_spp = f1_score(df_eval[spp_mask]['y_test'], df_eval[spp_mask]['y_pred'])
            gap_absoluto = abs(f1_concepcion - f1_spp)
            
            print(f"--- D3. Análisis de Fairness (Gap F1-Score) ---")
            print(f"  F1 Concepción: {f1_concepcion:.4f}")
            print(f"  F1 San Pedro de la Paz: {f1_spp:.4f}")
            print(f"  Gap Absoluto: {gap_absoluto:.4f}")
        else:
            print("--- D3. Análisis de Fairness (Gap F1-Score) ---")
            print("  No hay suficientes datos de CCP o SPP en el set de test para calcular el gap.")
    
    except Exception as e:
        print(f"  No se pudo calcular el Gap de Fairness: {e}")
        
    print("-----------------------------------------------------")