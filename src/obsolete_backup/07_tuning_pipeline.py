import os
import pandas as pd
import numpy as np
import json
import joblib
import time
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
import xgboost as xgb
import lightgbm as lgb

def ejecutar_tuning():
    print("Iniciando Pipeline de Tuning (Paso 7)...")
    
    # 1. Cargar datos
    print("Cargando datos...")
    X_train = pd.read_csv("data/processed/X_train_resampled.csv")
    y_train = pd.read_csv("data/processed/y_train_resampled.csv").values.ravel()
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()
    
    with open("results/label_mapping.json", "r") as f:
        label_mapping = json.load(f)
    class_names = [k for k, v in sorted(label_mapping.items(), key=lambda item: item[1])]
    
    # 2. Tomar muestra de tuning (10%)
    _, X_tune, _, y_tune = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
    )
    print(f"Muestra de entrenamiento para Tuning: {X_tune.shape[0]:,} registros")
    
    # 3. Tuning XGBoost en GPU
    print("\n--- Tuning XGBoost ---")
    xgb_base = xgb.XGBClassifier(
        tree_method='hist',
        device='cuda',
        random_state=42,
        n_jobs=-1
    )
    param_grid_xgb = {
        'n_estimators': [50, 100, 150],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.05, 0.1, 0.2],
        'min_child_weight': [1, 5, 10],
        'subsample': [0.8, 1.0]
    }
    xgb_search = RandomizedSearchCV(
        estimator=xgb_base,
        param_distributions=param_grid_xgb,
        n_iter=5,
        scoring='f1_macro',
        cv=3,
        random_state=42,
        verbose=1
    )
    start = time.time()
    xgb_search.fit(X_tune, y_tune)
    print(f"XGBoost Tuning completado en {time.time() - start:.2f} s.")
    best_params_xgb = xgb_search.best_params_
    print(f"Mejores parámetros XGBoost: {best_params_xgb}")
    
    # 4. Tuning LightGBM
    print("\n--- Tuning LightGBM ---")
    lgb_base = lgb.LGBMClassifier(
        random_state=42,
        n_jobs=-1,
        verbosity=-1
    )
    param_grid_lgb = {
        'n_estimators': [50, 100, 150],
        'max_depth': [4, 6, 8],
        'num_leaves': [20, 31, 50],
        'learning_rate': [0.05, 0.1, 0.2],
        'min_child_samples': [10, 20, 50]
    }
    lgb_search = RandomizedSearchCV(
        estimator=lgb_base,
        param_distributions=param_grid_lgb,
        n_iter=5,
        scoring='f1_macro',
        cv=3,
        random_state=42,
        verbose=1
    )
    start = time.time()
    lgb_search.fit(X_tune, y_tune)
    print(f"LightGBM Tuning completado en {time.time() - start:.2f} s.")
    best_params_lgb = lgb_search.best_params_
    print(f"Mejores parámetros LightGBM: {best_params_lgb}")
    
    # 5. Reentrenar Stacking Classifier Optimizado
    print("\n--- Reentrenando Stacking Optimizado (cv=3) ---")
    estimators_opt = [
        ('xgb', xgb.XGBClassifier(
            **best_params_xgb,
            tree_method='hist', device='cuda', random_state=42, n_jobs=-1
        )),
        ('lgb', lgb.LGBMClassifier(
            **best_params_lgb,
            random_state=42, n_jobs=-1, verbosity=-1
        ))
    ]
    stacking_opt = StackingClassifier(
        estimators=estimators_opt,
        final_estimator=LogisticRegression(max_iter=1000, random_state=42),
        cv=3,
        n_jobs=1
    )
    start = time.time()
    stacking_opt.fit(X_train, y_train)
    print(f"Stacking Optimizado entrenado en {time.time() - start:.2f} segundos.")
    joblib.dump(stacking_opt, "models/stacking_model_opt.joblib")
    
    # 6. Evaluación y Comparación
    print("\nEvaluando Stacking Optimizado...")
    preds_opt = stacking_opt.predict(X_test)
    acc_opt = accuracy_score(y_test, preds_opt)
    print(f"Accuracy Optimizado: {acc_opt*100:.4f}%")
    
    report_opt = classification_report(y_test, preds_opt, target_names=class_names, output_dict=True)
    df_report_opt = pd.DataFrame(report_opt).transpose()
    df_report_opt.to_csv("results/reporte_stacking_opt.csv")
    
    try:
        df_report_orig = pd.read_csv("results/reporte_stacking.csv", index_col=0)
        
        f1_comparison = pd.DataFrame({
            'Clase': class_names,
            'Stacking Original': [df_report_orig.loc[c, 'f1-score'] for c in class_names],
            'Stacking Optimizado': [df_report_opt.loc[c, 'f1-score'] for c in class_names]
        })
        f1_comparison['Mejora Absoluta'] = (f1_comparison['Stacking Optimizado'] - f1_comparison['Stacking Original']).round(4)
        print("\n--- Comparativa de F1-Score: Original vs. Optimizado ---")
        print(f1_comparison)
        
        f1_comparison.to_csv("results/comparativa_tuning_stacking.csv", index=False)
        
        # Graficar comparación
        df_melted = pd.melt(f1_comparison.drop(columns=['Mejora Absoluta']), id_vars=['Clase'], var_name='Modelo', value_name='F1-Score')
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df_melted, x='Clase', y='F1-Score', hue='Modelo', palette='Set1')
        plt.title("Comparación del F1-Score: Stacking Original vs. Optimizado", fontsize=14, fontweight='bold', pad=15)
        plt.xlabel("Categoría")
        plt.ylabel("F1-Score")
        plt.ylim(0.8, 1.02)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("results/figures/13_comparativa_tuning.png", dpi=300)
        plt.close()
        print("Gráfico de comparación guardado en results/figures/13_comparativa_tuning.png")
    except Exception as e:
        print(f"Error al generar comparativa: {e}")
        
    print("\n¡Pipeline de Tuning completado con éxito!")

if __name__ == "__main__":
    ejecutar_tuning()
