import os
import pandas as pd
import numpy as np
import json
import joblib
import time
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
import xgboost as xgb
import lightgbm as lgb

def ejecutar_modelado():
    print("Iniciando Pipeline de Modelado (Paso 6)...")
    
    # 1. Cargar datos
    print("Cargando datos...")
    X_train = pd.read_csv("data/processed/X_train_resampled.csv")
    y_train = pd.read_csv("data/processed/y_train_resampled.csv").values.ravel()
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()
    
    with open("results/label_mapping.json", "r") as f:
        label_mapping = json.load(f)
    # Ordenar por el valor entero para asegurar correspondencia
    class_names = [k for k, v in sorted(label_mapping.items(), key=lambda item: item[1])]
    print(f"Clases a evaluar: {class_names}")
    
    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # 2. Entrenar XGBoost en GPU
    print("\n--- Entrenando XGBoost en GPU ---")
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        tree_method='hist',
        device='cuda',
        random_state=42,
        n_jobs=-1
    )
    start = time.time()
    xgb_model.fit(X_train, y_train)
    xgb_time = time.time() - start
    print(f"XGBoost entrenado en {xgb_time:.2f} segundos.")
    xgb_model.save_model("models/xgboost_model.json")
    
    # 3. Entrenar LightGBM
    print("\n--- Entrenando LightGBM ---")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        num_leaves=31,
        random_state=42,
        n_jobs=-1,
        verbosity=-1
    )
    start = time.time()
    lgb_model.fit(X_train, y_train)
    lgb_time = time.time() - start
    print(f"LightGBM entrenado en {lgb_time:.2f} segundos.")
    lgb_model.booster_.save_model("models/lightgbm_model.txt")
    
    # 4. Entrenar Stacking Classifier (cv=3)
    print("\n--- Entrenando Stacking Classifier (Ensemble) ---")
    estimators = [
        ('xgb', xgb.XGBClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=6, 
            tree_method='hist', device='cuda', random_state=42, n_jobs=-1
        )),
        ('lgb', lgb.LGBMClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=6, 
            num_leaves=31, random_state=42, n_jobs=-1, verbosity=-1
        ))
    ]
    stacking_model = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000, random_state=42),
        cv=3,
        n_jobs=1, # Ejecución secuencial para no sobrecargar la GPU en cross-validation
        passthrough=False
    )
    start = time.time()
    stacking_model.fit(X_train, y_train)
    stack_time = time.time() - start
    print(f"Stacking Classifier entrenado en {stack_time:.2f} segundos.")
    joblib.dump(stacking_model, "models/stacking_model.joblib")
    
    # 5. Evaluación y métricas
    def evaluar(model, name):
        print(f"\nEvaluating: {name}")
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"Accuracy: {acc*100:.4f}%")
        
        report = classification_report(y_test, preds, target_names=class_names, output_dict=True)
        df_report = pd.DataFrame(report).transpose()
        df_report.to_csv(f"results/reporte_{name.lower()}.csv")
        
        print(classification_report(y_test, preds, target_names=class_names))
        
        cm = confusion_matrix(y_test, preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
        plt.title(f"Matriz de Confusión - {name}", fontsize=14, fontweight='bold', pad=15)
        plt.ylabel("Clase Real")
        plt.xlabel("Clase Predicha")
        plt.tight_layout()
        plt.savefig(f"results/figures/matriz_confusion_{name.lower()}.png", dpi=300)
        plt.close()
        return df_report

    report_xgb = evaluar(xgb_model, "XGBoost")
    report_lgb = evaluar(lgb_model, "LightGBM")
    report_stack = evaluar(stacking_model, "Stacking")
    
    # 6. Comparar F1-Score
    f1_comparison = pd.DataFrame({
        'Clase': class_names,
        'XGBoost': [report_xgb.loc[c, 'f1-score'] for c in class_names],
        'LightGBM': [report_lgb.loc[c, 'f1-score'] for c in class_names],
        'Stacking': [report_stack.loc[c, 'f1-score'] for c in class_names]
    })
    f1_comparison.to_csv("results/comparativa_f1_score.csv", index=False)
    print("\nComparativa de F1-Score:")
    print(f1_comparison.round(4))
    
    # Graficar comparativa F1
    df_melted = pd.melt(f1_comparison, id_vars=['Clase'], var_name='Modelo', value_name='F1-Score')
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_melted, x='Clase', y='F1-Score', hue='Modelo', palette='Set2')
    plt.title("Comparación del F1-Score por Clase y Modelo", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Categoría")
    plt.ylabel("F1-Score")
    plt.ylim(0.8, 1.02)
    plt.xticks(rotation=45)
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig("results/figures/11_comparativa_f1_score.png", dpi=300)
    plt.close()
    
    # 7. Importancia de características
    features = X_train.columns
    xgb_importances = xgb_model.feature_importances_
    lgb_importances = lgb_model.feature_importances_ / lgb_model.feature_importances_.sum()
    
    df_importance = pd.DataFrame({
        'Característica': features,
        'XGBoost': xgb_importances,
        'LightGBM': lgb_importances
    })
    df_importance['Media'] = df_importance[['XGBoost', 'LightGBM']].mean(axis=1)
    df_importance = df_importance.sort_values(by='Media', ascending=False).head(10)
    df_importance.to_csv("results/importancia_caracteristicas.csv", index=False)
    
    # Graficar importancia
    df_imp_melted = pd.melt(df_importance.drop(columns=['Media']), id_vars=['Característica'], var_name='Modelo', value_name='Importancia')
    plt.figure(figsize=(12, 7))
    sns.barplot(data=df_imp_melted, x='Importancia', y='Característica', hue='Modelo', palette='viridis')
    plt.title("Top 10 Características más Determinantes en el Modelo Híbrido", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Importancia Relativa")
    plt.ylabel("Característica")
    plt.tight_layout()
    plt.savefig("results/figures/12_importancia_caracteristicas.png", dpi=300)
    plt.close()
    
    print("\n¡Pipeline de modelado completado con éxito!")

if __name__ == "__main__":
    ejecutar_modelado()
