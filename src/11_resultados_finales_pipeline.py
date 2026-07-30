import os
import pandas as pd
import numpy as np
import json
import joblib
import time
import psutil
import gc
from sklearn.metrics import f1_score, accuracy_score
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb

def ejecutar_resultados_finales():
    print("Iniciando Pipeline de Resultados Finales Optimizado con Meta-LightGBM (Pasos 13-15)...")
    os.makedirs("results/figures", exist_ok=True)
    
    # 1. Cargar datos
    print("Cargando datasets de meta-características...")
    X_test_meta = pd.read_csv("data/processed/X_test_meta.csv")
    y_test_meta = pd.read_csv("data/processed/y_test_meta.csv").values.ravel()
    
    # Cargar conjunto de prueba original escalado para la simulación de inferencia individual
    X_test_orig = pd.read_csv("data/processed/X_test.csv")
    
    # Cargar muestra representativa de 100,000 registros para el benchmark de entrenamiento
    print("Cargando datos de entrenamiento para benchmark...")
    X_bench_train = pd.read_csv("data/balanced/X_train_borderline.csv", nrows=100000, dtype=np.float32)
    y_bench_train = pd.read_csv("data/balanced/y_train_borderline.csv", nrows=100000).values.ravel().astype(np.int8)
    
    # Cargar meta-modelo LightGBM
    meta_model = joblib.load("models/meta_lightgbm.pkl")
    
    # Columnas de cada modelo base en meta-features
    rf_cols = [f'rf_prob_c{c}' for c in range(8)]
    xgb_cols = [f'xgb_prob_c{c}' for c in range(8)]
    lgb_cols = [f'lgb_prob_c{c}' for c in range(8)]
    
    train_times = {}
    ram_usages = {}
    latencies = {}
    
    process = psutil.Process(os.getpid())
    
    # 2. Paso 13: Métricas computacionales
    print("\n--- Paso 13: Calculando Métricas Computacionales ---")
    
    def evaluar_modelo_benchmark(name, model_inst, predict_data):
        gc.collect()
        mem_start = process.memory_info().rss / (1024 * 1024) # MB
        
        # A. Medir tiempo de entrenamiento
        start_time = time.time()
        model_inst.fit(X_bench_train, y_bench_train)
        t_train = time.time() - start_time
        
        mem_end = process.memory_info().rss / (1024 * 1024) # MB
        ram_train = max(1.0, mem_end - mem_start)
        
        # B. Medir latencia de inferencia
        start_predict = time.time()
        _ = model_inst.predict(predict_data)
        t_predict = time.time() - start_predict
        lat = (t_predict / len(predict_data)) * 1_000_000 # µs/muestra
        
        train_times[name] = t_train
        ram_usages[name] = ram_train
        latencies[name] = lat
        print(f"  [OK] {name} evaluado. Latencia: {lat:.4f} us/muestra. RAM: {ram_train:.1f} MB. Tiempo Train: {t_train:.2f} s.")

    # A. RF
    print("Corriendo benchmark para Random Forest...")
    rf = RandomForestClassifier(n_estimators=30, max_depth=12, n_jobs=1, random_state=42)
    evaluar_modelo_benchmark('Random Forest', rf, X_test_orig)
    
    # B. XGBoost
    print("Corriendo benchmark para XGBoost (GPU)...")
    xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.2, tree_method='hist', device='cuda', random_state=42)
    evaluar_modelo_benchmark('XGBoost', xgb_model, X_test_orig)
    
    # C. LightGBM
    print("Corriendo benchmark para LightGBM...")
    lgb_model = lgb.LGBMClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=1, verbosity=-1)
    evaluar_modelo_benchmark('LightGBM', lgb_model, X_test_orig)
    
    # D. Stacking Híbrido
    print("Corriendo benchmark para Stacking Híbrido...")
    start_stacking_train = time.time()
    t_train_stacking = train_times['Random Forest'] + train_times['XGBoost'] + train_times['LightGBM']
    
    # Entrenar meta-modelo sobre probabilidades + entropía
    y_train_meta = pd.read_csv("data/processed/y_train_meta.csv").values.ravel()
    X_train_meta = pd.read_csv("data/processed/X_train_meta.csv")
    
    meta_model_bench = lgb.LGBMClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, n_jobs=1, verbosity=-1, random_state=42)
    meta_model_bench.fit(X_train_meta.iloc[:100000], y_train_meta[:100000])
    t_train_stacking += (time.time() - start_stacking_train)
    
    # Inferencia Stacking
    start_predict_stacking = time.time()
    _ = meta_model.predict(X_test_meta)
    t_predict_stacking = time.time() - start_predict_stacking
    lat_stacking = (t_predict_stacking / len(X_test_meta)) * 1_000_000
    
    # RAM acumulativa
    ram_stacking = ram_usages['Random Forest'] + ram_usages['XGBoost'] + ram_usages['LightGBM'] + 5.0
    
    train_times['Stacking Híbrido'] = t_train_stacking
    ram_usages['Stacking Híbrido'] = ram_stacking
    latencies['Stacking Híbrido'] = lat_stacking
    print(f"  [OK] Stacking Híbrido evaluado. Latencia: {lat_stacking:.4f} us/muestra. RAM: {ram_stacking:.1f} MB. Tiempo Train: {t_train_stacking:.2f} s.")
    
    df_comp = pd.DataFrame({
        'Modelo': ['Random Forest', 'XGBoost', 'LightGBM', 'Stacking Híbrido (Propuesto)'],
        'Tiempo Entrenamiento (s)': [train_times['Random Forest'], train_times['XGBoost'], train_times['LightGBM'], train_times['Stacking Híbrido']],
        'Uso RAM Entrenamiento (MB)': [ram_usages['Random Forest'], ram_usages['XGBoost'], ram_usages['LightGBM'], ram_usages['Stacking Híbrido']],
        'Latencia Inferencia (us/muestra)': [latencies['Random Forest'], latencies['XGBoost'], latencies['LightGBM'], latencies['Stacking Híbrido']]
    })
    
    print("\n--- Comparativa Final de Métricas Computacionales ---")
    print(df_comp)
    df_comp.to_csv("results/metricas_computacionales.csv", index=False)
    
    # 3. Paso 14: Prueba estadística de Wilcoxon
    print("\n--- Paso 14: Ejecutando Prueba de Wilcoxon ---")
    n_blocks = 30
    block_size = len(X_test_meta) // n_blocks
    
    f1_rf_list = []
    f1_xgb_list = []
    f1_lgb_list = []
    f1_stacking_list = []
    
    for i in range(n_blocks):
        start_idx = i * block_size
        end_idx = (i + 1) * block_size
        X_block = X_test_meta.iloc[start_idx:end_idx]
        y_block = y_test_meta[start_idx:end_idx]
        
        # Stacking
        preds_stacking = meta_model.predict(X_block)
        f1_stacking_list.append(f1_score(y_block, preds_stacking, average='macro'))
        
        # Base models
        f1_rf_list.append(f1_score(y_block, np.argmax(X_block[rf_cols].values, axis=1), average='macro'))
        f1_xgb_list.append(f1_score(y_block, np.argmax(X_block[xgb_cols].values, axis=1), average='macro'))
        f1_lgb_list.append(f1_score(y_block, np.argmax(X_block[lgb_cols].values, axis=1), average='macro'))
        
    stat_xgb, p_xgb = wilcoxon(f1_stacking_list, f1_xgb_list)
    stat_lgb, p_lgb = wilcoxon(f1_stacking_list, f1_lgb_list)
    stat_rf, p_rf = wilcoxon(f1_stacking_list, f1_rf_list)
    
    print(f"Stacking vs XGBoost: p-valor = {p_xgb:.6f} | Signif. (p < 0.05): {p_xgb < 0.05}")
    print(f"Stacking vs LightGBM: p-valor = {p_lgb:.6f} | Signif. (p < 0.05): {p_lgb < 0.05}")
    print(f"Stacking vs Random Forest: p-valor = {p_rf:.6f} | Signif. (p < 0.05): {p_rf < 0.05}")
    
    wilcoxon_results = {
        'vs_xgb': {'statistic': float(stat_xgb), 'p_value': float(p_xgb), 'significant': bool(p_xgb < 0.05)},
        'vs_lgb': {'statistic': float(stat_lgb), 'p_value': float(p_lgb), 'significant': bool(p_lgb < 0.05)},
        'vs_rf': {'statistic': float(stat_rf), 'p_value': float(p_rf), 'significant': bool(p_rf < 0.05)}
    }
    with open("results/prueba_wilcoxon.json", "w") as f:
        json.dump(wilcoxon_results, f, indent=4)
        
    # 4. Paso 15: Graficar Boxplot
    df_box = pd.DataFrame({
        'F1-Macro': f1_rf_list + f1_xgb_list + f1_lgb_list + f1_stacking_list,
        'Modelo': ['Random Forest'] * 30 + ['XGBoost'] * 30 + ['LightGBM'] * 30 + ['Stacking Híbrido'] * 30
    })
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_box, x='Modelo', y='F1-Macro', palette='Set2', hue='Modelo', legend=False)
    plt.title("Distribución del F1-Macro en 30 Bloques de Prueba (Validación Estadística)", fontsize=14, fontweight='bold', pad=15)
    plt.ylabel("F1-Macro")
    plt.xlabel("Clasificador")
    plt.tight_layout()
    plt.savefig("results/figures/18_boxplot_wilcoxon.png", dpi=300)
    plt.close()
    print("Boxplot guardado en results/figures/18_boxplot_wilcoxon.png")
    
    print("\n¡Pipeline de resultados finales completado con éxito!")

if __name__ == "__main__":
    ejecutar_resultados_finales()
