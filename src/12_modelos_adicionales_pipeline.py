import os
import pandas as pd
import numpy as np
import json
import joblib
import time
import gc
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score

def ejecutar_modelos_adicionales():
    print("Iniciando Pipeline de Modelos Adicionales para Comparación Completa...")
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # 1. Cargar datos
    print("Cargando conjuntos de datos (float32)...")
    X_train_full = pd.read_csv("data/balanced/X_train_borderline.csv", dtype=np.float32)
    y_train_full = pd.read_csv("data/balanced/y_train_borderline.csv").values.ravel().astype(np.int8)
    
    X_test = pd.read_csv("data/processed/X_test.csv", dtype=np.float32)
    y_test = pd.read_csv("data/processed/y_test.csv").values.ravel().astype(np.int8)
    
    with open("results/label_mapping.json", "r") as f:
        label_mapping = json.load(f)
    class_names = [k for k, v in sorted(label_mapping.items(), key=lambda item: item[1])]
    
    # 2. Tomar la muestra representativa del 25% (exactamente igual que los base)
    print("Tomando muestra estratificada del 25%...")
    _, X_train, _, y_train = train_test_split(
        X_train_full, y_train_full, test_size=0.25, random_state=42, stratify=y_train_full
    )
    
    del X_train_full, y_train_full
    gc.collect()
    
    print(f"Muestras de entrenamiento: {X_train.shape[0]:,}")
    print(f"Muestras de prueba: {X_test.shape[0]:,}")
    
    dict_resultados = {}
    
    # --- MODELO A: Árbol de Decisión CART (Línea Base Tradicional) ---
    print("\n1. Entrenando Árbol de Decisión (CART)...")
    start = time.time()
    dt = DecisionTreeClassifier(max_depth=12, random_state=42)
    dt.fit(X_train, y_train)
    t_train = time.time() - start
    
    print("Evaluando CART...")
    preds = dt.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1_classes = f1_score(y_test, preds, average=None)
    f1_macro = f1_score(y_test, preds, average='macro')
    
    dict_resultados['CART'] = {
        'Accuracy': acc,
        'F1_Macro': f1_macro,
        'Tiempo_Train': t_train,
        'F1_Clases': f1_classes
    }
    joblib.dump(dt, "models/baseline_dt.pkl")
    del dt, preds
    gc.collect()
    
    # --- MODELO B: Regresión Logística (Línea Base Clásica) ---
    print("\n2. Entrenando Regresión Logística (LBFGS)...")
    start = time.time()
    lr = LogisticRegression(max_iter=100, solver='lbfgs', multi_class='multinomial', random_state=42)
    lr.fit(X_train, y_train)
    t_train = time.time() - start
    
    print("Evaluando Regresión Logística...")
    preds = lr.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1_classes = f1_score(y_test, preds, average=None)
    f1_macro = f1_score(y_test, preds, average='macro')
    
    dict_resultados['Regresión Logística'] = {
        'Accuracy': acc,
        'F1_Macro': f1_macro,
        'Tiempo_Train': t_train,
        'F1_Clases': f1_classes
    }
    joblib.dump(lr, "models/baseline_lr.pkl")
    del lr, preds
    gc.collect()
    
    # --- MODELO C: Perceptrón Multicapa (MLP / Red Neuronal Clásica) ---
    print("\n3. Entrenando Red Neuronal (MLP Classifier)...")
    start = time.time()
    # Una arquitectura simple (32, 16) con max_iter=10 entrena muy rápido (menos de 20s) y evita OOM
    mlp = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=10, random_state=42)
    mlp.fit(X_train, y_train)
    t_train = time.time() - start
    
    print("Evaluando MLP...")
    preds = mlp.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1_classes = f1_score(y_test, preds, average=None)
    f1_macro = f1_score(y_test, preds, average='macro')
    
    dict_resultados['Red Neuronal (MLP)'] = {
        'Accuracy': acc,
        'F1_Macro': f1_macro,
        'Tiempo_Train': t_train,
        'F1_Clases': f1_classes
    }
    joblib.dump(mlp, "models/baseline_mlp.pkl")
    del mlp, preds
    gc.collect()
    
    # --- 4. CARGAR RESULTADOS DE LOS MODELOS ANTERIORES PARA LA TABLA COMPARATIVA ---
    print("\nCargando resultados de los modelos existentes...")
    
    # Cargar modelos base entrenados para evaluarlos en X_test
    # Random Forest
    rf = joblib.load("models/base_rf.pkl")
    preds = rf.predict(X_test)
    dict_resultados['Random Forest'] = {
        'Accuracy': accuracy_score(y_test, preds),
        'F1_Macro': f1_score(y_test, preds, average='macro'),
        'Tiempo_Train': 2.03, # Referencia del benchmark anterior
        'F1_Clases': f1_score(y_test, preds, average=None)
    }
    del rf, preds
    gc.collect()
    
    # XGBoost
    xgb_model = joblib.load("models/base_xgb.pkl")
    preds = xgb_model.predict(X_test)
    dict_resultados['XGBoost'] = {
        'Accuracy': accuracy_score(y_test, preds),
        'F1_Macro': f1_score(y_test, preds, average='macro'),
        'Tiempo_Train': 3.00,
        'F1_Clases': f1_score(y_test, preds, average=None)
    }
    del xgb_model, preds
    gc.collect()
    
    # LightGBM
    lgb_model = joblib.load("models/base_lgb.pkl")
    preds = lgb_model.predict(X_test)
    dict_resultados['LightGBM'] = {
        'Accuracy': accuracy_score(y_test, preds),
        'F1_Macro': f1_score(y_test, preds, average='macro'),
        'Tiempo_Train': 4.17,
        'F1_Clases': f1_score(y_test, preds, average=None)
    }
    del lgb_model, preds
    gc.collect()
    
    # Stacking Híbrido (Cargar reporte ya guardado)
    df_stacking = pd.read_csv("results/reporte_stacking_hibrido.csv", index_col=0)
    # Extraer F1-scores de clases
    f1_stack_clases = []
    for cname in class_names:
        f1_stack_clases.append(df_stacking.loc[cname, 'f1-score'])
    
    dict_resultados['Stacking Híbrido (Propuesto)'] = {
        'Accuracy': df_stacking.loc['accuracy', 'precision'],
        'F1_Macro': df_stacking.loc['macro avg', 'f1-score'],
        'Tiempo_Train': 11.53,
        'F1_Clases': np.array(f1_stack_clases)
    }
    
    # 5. Generar DataFrame Consolidado de F1-Scores por clase
    print("\n--- Generando Comparativa Final de F1-Scores por Clase ---")
    data_f1 = []
    for name, res in dict_resultados.items():
        row = {'Modelo': name}
        for i, cname in enumerate(class_names):
            row[cname] = round(res['F1_Clases'][i], 4)
        row['F1-Macro Promedio'] = round(res['F1_Macro'], 4)
        row['Accuracy Global'] = round(res['Accuracy'] * 100, 2)
        data_f1.append(row)
        
    df_final = pd.DataFrame(data_f1)
    print(df_final)
    df_final.to_csv("results/reporte_comparativa_completa.csv", index=False)
    print("\nComparativa guardada en 'results/reporte_comparativa_completa.csv'")
    
    print("\n¡Pipeline de comparación de modelos adicionales completado con éxito!")

if __name__ == "__main__":
    ejecutar_modelos_adicionales()
