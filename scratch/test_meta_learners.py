import os
import pandas as pd
import numpy as np
import json
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from sklearn.metrics import classification_report, f1_score, accuracy_score

def test_meta_learners():
    print("Iniciando Experimento de Meta-Modelos Alternativos...")
    
    # 1. Cargar meta-features
    X_train_meta = pd.read_csv("data/processed/X_train_meta.csv")
    y_train_meta = pd.read_csv("data/processed/y_train_meta.csv").values.ravel()
    X_test_meta = pd.read_csv("data/processed/X_test_meta.csv")
    y_test_meta = pd.read_csv("data/processed/y_test_meta.csv").values.ravel()
    
    with open("results/label_mapping.json", "r") as f:
        label_mapping = json.load(f)
    class_names = [k for k, v in sorted(label_mapping.items(), key=lambda item: item[1])]
    
    # Cargar scores de XGBoost base para referencia
    xgb_f1s = [0.6238, 0.4149, 0.9006, 0.6986, 0.9978, 0.8218, 0.8076, 0.3027]
    
    meta_learners = {
        'CART (Actual)': DecisionTreeClassifier(criterion='gini', max_depth=10, min_samples_leaf=100, random_state=42),
        'CART (Más Profundo)': DecisionTreeClassifier(criterion='gini', max_depth=15, min_samples_leaf=50, random_state=42),
        'Random Forest Meta': RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_leaf=50, n_jobs=-1, random_state=42),
        'Logistic Regression Meta': LogisticRegression(max_iter=500, solver='lbfgs', random_state=42),
        'LightGBM Meta': lgb.LGBMClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, n_jobs=-1, random_state=42, verbosity=-1)
    }
    
    for name, clf in meta_learners.items():
        print(f"\n--- Probando Meta-Modelo: {name} ---")
        clf.fit(X_train_meta, y_train_meta)
        preds = clf.predict(X_test_meta)
        
        acc = accuracy_score(y_test_meta, preds)
        macro_f1 = f1_score(y_test_meta, preds, average='macro')
        f1_classes = f1_score(y_test_meta, preds, average=None)
        
        print(f"Accuracy: {acc*100:.2f}% | F1-Macro: {macro_f1:.4f}")
        print("F1-Scores por Clase:")
        for i, cname in enumerate(class_names):
            diff_vs_xgb = f1_classes[i] - xgb_f1s[i]
            sign = "+" if diff_vs_xgb >= 0 else ""
            print(f"  {cname:12}: {f1_classes[i]:.4f} (vs XGBoost: {sign}{diff_vs_xgb*100:.2f}%)")

if __name__ == "__main__":
    test_meta_learners()
