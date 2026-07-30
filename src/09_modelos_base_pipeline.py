import os
import pandas as pd
import numpy as np
import json
import joblib
import time
import gc
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb
import lightgbm as lgb

def calcular_entropia_shannon(probabilidades):
    eps = 1e-15
    probabilidades = np.clip(probabilidades, eps, 1.0)
    return -np.sum(probabilidades * np.log2(probabilidades), axis=1)

def crear_meta_features(probs_rf, probs_xgb, probs_lgb, ent_rf, ent_xgb, ent_lgb):
    meta_df = pd.DataFrame()
    for c in range(8):
        meta_df[f'rf_prob_c{c}'] = probs_rf[:, c]
        meta_df[f'xgb_prob_c{c}'] = probs_xgb[:, c]
        meta_df[f'lgb_prob_c{c}'] = probs_lgb[:, c]
    meta_df['rf_entropy'] = ent_rf
    meta_df['xgb_entropy'] = ent_xgb
    meta_df['lgb_entropy'] = ent_lgb
    return meta_df

def ejecutar_modelos_base():
    print("Iniciando Pipeline de Modelos Base Optimizado en RAM (Previene Caídas de WSL)...")
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # 1. Cargar datos usando float32 (reduce la memoria a la mitad)
    print("Cargando particiones del conjunto de datos en float32...")
    X_train_full = pd.read_csv("data/balanced/X_train_borderline.csv", dtype=np.float32)
    y_train_full = pd.read_csv("data/balanced/y_train_borderline.csv").values.ravel().astype(np.int8)
    
    X_val = pd.read_csv("data/processed/X_val.csv", dtype=np.float32)
    y_val = pd.read_csv("data/processed/y_val.csv").values.ravel().astype(np.int8)
    
    X_test = pd.read_csv("data/processed/X_test.csv", dtype=np.float32)
    y_test = pd.read_csv("data/processed/y_test.csv").values.ravel().astype(np.int8)
    
    with open("results/label_mapping.json", "r") as f:
        label_mapping = json.load(f)
    class_names = [k for k, v in sorted(label_mapping.items(), key=lambda item: item[1])]
    
    print(f"Full Train balanceado: {X_train_full.shape[0]:,} muestras.")
    
    # 2. Tomar una muestra representativa del 25% del conjunto de entrenamiento
    # Esto reduce los requisitos de RAM en un 75% manteniendo una gran robustez estadística.
    print("Tomando muestra estratificada del 25% para evitar caídas por falta de memoria (OOM)...")
    _, X_train, _, y_train = train_test_split(
        X_train_full, y_train_full, test_size=0.25, random_state=42, stratify=y_train_full
    )
    
    # Liberar el dataset completo de la memoria
    del X_train_full, y_train_full
    gc.collect()
    
    print(f"Entrenamiento ajustado: {X_train.shape[0]:,} muestras, {X_train.shape[1]} características.")
    print(f"Validación: {X_val.shape[0]:,} muestras.")
    print(f"Prueba: {X_test.shape[0]:,} muestras.")
    
    # 3. Loop de validación cruzada (3 pliegues)
    oof_probs_rf = np.zeros((X_train.shape[0], 8))
    oof_probs_xgb = np.zeros((X_train.shape[0], 8))
    oof_probs_lgb = np.zeros((X_train.shape[0], 8))
    
    val_probs_rf = np.zeros((X_val.shape[0], 8))
    val_probs_xgb = np.zeros((X_val.shape[0], 8))
    val_probs_lgb = np.zeros((X_val.shape[0], 8))
    
    test_probs_rf = np.zeros((X_test.shape[0], 8))
    test_probs_xgb = np.zeros((X_test.shape[0], 8))
    test_probs_lgb = np.zeros((X_test.shape[0], 8))
    
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        start_fold = time.time()
        print(f"\n========== ENTRENANDO PLIEGUE {fold+1} / 3 ==========")
        
        X_tr = X_train.iloc[train_idx]
        y_tr = y_train[train_idx]
        X_va_fold = X_train.iloc[val_idx]
        
        # A. Random Forest (n_jobs=1 secuencial para evitar forks de Loky que provocan crash)
        print("  -> Entrenando Random Forest (n_jobs=1, secuencial)...")
        rf = RandomForestClassifier(n_estimators=30, max_depth=12, n_jobs=1, random_state=42)
        rf.fit(X_tr, y_tr)
        oof_probs_rf[val_idx] = rf.predict_proba(X_va_fold)
        val_probs_rf += rf.predict_proba(X_val) / 3
        test_probs_rf += rf.predict_proba(X_test) / 3
        
        # B. XGBoost GPU (Usa memoria GPU, muy bajo consumo en CPU)
        print("  -> Entrenando XGBoost (GPU)...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.2,
            tree_method='hist', device='cuda', random_state=42
        )
        xgb_model.fit(X_tr, y_tr)
        oof_probs_xgb[val_idx] = xgb_model.predict_proba(X_va_fold)
        val_probs_xgb += xgb_model.predict_proba(X_val) / 3
        test_probs_xgb += xgb_model.predict_proba(X_test) / 3
        
        # C. LightGBM (n_jobs=1 para estabilidad extrema)
        print("  -> Entrenando LightGBM (n_jobs=1)...")
        lgb_model = lgb.LGBMClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=42, n_jobs=1, verbosity=-1
        )
        lgb_model.fit(X_tr, y_tr)
        oof_probs_lgb[val_idx] = lgb_model.predict_proba(X_va_fold)
        val_probs_lgb += lgb_model.predict_proba(X_val) / 3
        test_probs_lgb += lgb_model.predict_proba(X_test) / 3
        
        if fold == 0:
            joblib.dump(rf, "models/base_rf.pkl")
            joblib.dump(xgb_model, "models/base_xgb.pkl")
            joblib.dump(lgb_model, "models/base_lgb.pkl")
            print("  [Info] Modelos base guardados en 'models/'")
            
        # Liberar variables temporales y forzar GC
        del X_tr, y_tr, X_va_fold
        if fold > 0:
            del rf, xgb_model, lgb_model
        gc.collect()
        
        print(f"  Pliegue {fold+1} finalizado en {time.time() - start_fold:.2f} s.")
        
    # 3. Calcular Entropías
    print("\nCalculando Entropías de Shannon...")
    entropy_tr_rf = calcular_entropia_shannon(oof_probs_rf)
    entropy_tr_xgb = calcular_entropia_shannon(oof_probs_xgb)
    entropy_tr_lgb = calcular_entropia_shannon(oof_probs_lgb)
    
    entropy_val_rf = calcular_entropia_shannon(val_probs_rf)
    entropy_val_xgb = calcular_entropia_shannon(val_probs_xgb)
    entropy_val_lgb = calcular_entropia_shannon(val_probs_lgb)
    
    entropy_te_rf = calcular_entropia_shannon(test_probs_rf)
    entropy_te_xgb = calcular_entropia_shannon(test_probs_xgb)
    entropy_te_lgb = calcular_entropia_shannon(test_probs_lgb)
    
    # 4. Crear Meta-Features
    print("\nCreando dataframes de Meta-Features...")
    X_train_meta = crear_meta_features(oof_probs_rf, oof_probs_xgb, oof_probs_lgb, entropy_tr_rf, entropy_tr_xgb, entropy_tr_lgb)
    X_val_meta = crear_meta_features(val_probs_rf, val_probs_xgb, val_probs_lgb, entropy_val_rf, entropy_val_xgb, entropy_val_lgb)
    X_test_meta = crear_meta_features(test_probs_rf, test_probs_xgb, test_probs_lgb, entropy_te_rf, entropy_te_xgb, entropy_te_lgb)
    
    X_train_meta.to_csv("data/processed/X_train_meta.csv", index=False)
    pd.DataFrame(y_train, columns=['Label']).to_csv("data/processed/y_train_meta.csv", index=False)
    
    X_val_meta.to_csv("data/processed/X_val_meta.csv", index=False)
    pd.DataFrame(y_val, columns=['Label']).to_csv("data/processed/y_val_meta.csv", index=False)
    
    X_test_meta.to_csv("data/processed/X_test_meta.csv", index=False)
    pd.DataFrame(y_test, columns=['Label']).to_csv("data/processed/y_test_meta.csv", index=False)
    
    print("Meta-features guardadas con éxito en 'data/processed/'.")
    
    # 5. Reporte de Validación
    for name, probs in [('Random Forest', val_probs_rf), ('XGBoost', val_probs_xgb), ('LightGBM', val_probs_lgb)]:
        preds = np.argmax(probs, axis=1)
        acc = accuracy_score(y_val, preds)
        print(f"\n--- Reporte de Validación: {name} (Accuracy: {acc*100:.2f}%) ---")
        print(classification_report(y_val, preds, target_names=class_names))

if __name__ == "__main__":
    ejecutar_modelos_base()
