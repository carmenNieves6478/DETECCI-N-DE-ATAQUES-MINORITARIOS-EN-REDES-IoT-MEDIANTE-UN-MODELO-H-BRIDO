import os
import pandas as pd
import numpy as np
import json
import joblib
import time
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

def ejecutar_meta_modelo():
    print("Iniciando Pipeline del Meta-Modelo LightGBM (Paso 11)...")
    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # 1. Cargar meta-features
    print("Cargando datasets de meta-características...")
    X_train_meta = pd.read_csv("data/processed/X_train_meta.csv")
    y_train_meta = pd.read_csv("data/processed/y_train_meta.csv").values.ravel()
    X_val_meta = pd.read_csv("data/processed/X_val_meta.csv")
    y_val_meta = pd.read_csv("data/processed/y_val_meta.csv").values.ravel()
    X_test_meta = pd.read_csv("data/processed/X_test_meta.csv")
    y_test_meta = pd.read_csv("data/processed/y_test_meta.csv").values.ravel()
    
    with open("results/label_mapping.json", "r") as f:
        label_mapping = json.load(f)
    class_names = [k for k, v in sorted(label_mapping.items(), key=lambda item: item[1])]
    
    print(f"Entrenamiento: {X_train_meta.shape[0]:,}, Validación: {X_val_meta.shape[0]:,}, Prueba: {X_test_meta.shape[0]:,}")
    
    # 2. Grid Search para el meta-modelo LightGBM
    print("\n--- Paso 11: Optimización del Meta-Clasificador LightGBM ---")
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [3, 4, 6],
        'learning_rate': [0.01, 0.05, 0.1],
        'verbosity': [-1]
    }
    
    meta_lgb = lgb.LGBMClassifier(random_state=42, n_jobs=2)
    grid_search = GridSearchCV(
        estimator=meta_lgb,
        param_grid=param_grid,
        scoring='f1_macro',
        cv=3,
        verbose=1
    )
    
    start_time = time.time()
    grid_search.fit(X_train_meta, y_train_meta)
    print(f"Búsqueda completada en {time.time() - start_time:.2f} segundos.")
    print(f"Mejores parámetros: {grid_search.best_params_}")
    
    best_meta_lgb = grid_search.best_estimator_
    joblib.dump(best_meta_lgb, "models/meta_lightgbm.pkl")
    print("Meta-modelo LightGBM guardado en models/meta_lightgbm.pkl")
    
    # 3. Evaluación en conjunto de prueba (Paso 12)
    print("\n--- Paso 12: Evaluación del Stacking Híbrido en Prueba ---")
    test_preds = best_meta_lgb.predict(X_test_meta)
    acc = accuracy_score(y_test_meta, test_preds)
    print(f"Accuracy del Stacking Híbrido: {acc*100:.4f}%")
    
    report = classification_report(y_test_meta, test_preds, target_names=class_names, output_dict=True)
    print("\nReporte de Clasificación del Stacking Híbrido (Prueba):")
    print(classification_report(y_test_meta, test_preds, target_names=class_names))
    
    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv("results/reporte_stacking_hibrido.csv")
    print("Reporte guardado en results/reporte_stacking_hibrido.csv")
    
    # 4. Matriz de confusión
    cm = confusion_matrix(y_test_meta, test_preds)
    plt.figure(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues', values_format='d', ax=plt.gca(), xticks_rotation=45)
    plt.title("Matriz de Confusión - Stacking Híbrido Propuesto (Meta-LightGBM)", fontsize=14, fontweight='bold', pad=15)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig("results/figures/16_matriz_confusion_stacking_hibrido.png", dpi=300)
    plt.close()
    print("Matriz de confusión guardada en results/figures/16_matriz_confusion_stacking_hibrido.png")
    
    # 5. Importancia de características del meta-modelo
    importances = best_meta_lgb.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=importances[indices[:15]], y=X_train_meta.columns[indices[:15]], palette='viridis')
    plt.title("Importancia de Meta-Características en el Meta-Clasificador (Top 15)", fontsize=14, fontweight='bold')
    plt.xlabel("Importancia Relativa (Gain / Split count)")
    plt.tight_layout()
    plt.savefig("results/figures/17_importancia_meta_modelo.png", dpi=300)
    plt.close()
    print("Gráfico de importancia guardado en results/figures/17_importancia_meta_modelo.png")
    
    print("\n¡Pipeline del Meta-Modelo completado con éxito!")

if __name__ == "__main__":
    ejecutar_meta_modelo()
