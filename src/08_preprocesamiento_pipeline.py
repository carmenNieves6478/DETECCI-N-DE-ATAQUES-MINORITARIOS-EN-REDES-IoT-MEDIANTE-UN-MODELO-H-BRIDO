import os
import pandas as pd
import numpy as np
import json
import joblib
import time
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import BorderlineSMOTE

def ejecutar_preprocesamiento():
    print("Iniciando Pipeline de Preprocesamiento y Borderline-SMOTE...")
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/balanced", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # 1. Cargar el dataset consolidado
    df_path = "data/processed/dataset_consolidado.csv"
    print("Cargando dataset consolidado...")
    df = pd.read_csv(df_path)
    print(f"Dataset cargado. Filas iniciales: {len(df):,}")
    
    # 2. Limpieza de datos (Paso 3)
    print("\n--- Paso 3: Limpieza de Datos ---")
    # A. Eliminar columnas redundantes (AVG, Tot sum, Min)
    cols_redundantes = ['AVG', 'Tot sum', 'Min']
    cols_a_eliminar = [c for c in cols_redundantes if c in df.columns]
    print(f"Eliminando columnas redundantes: {cols_a_eliminar}")
    df.drop(columns=cols_a_eliminar, inplace=True)
    
    # B. Eliminar filas con infinitos en 'Rate'
    if 'Rate' in df.columns:
        df = df[~np.isinf(df['Rate'])]
        print("Filas con valores infinitos en 'Rate' eliminadas.")
        
    # C. Eliminar filas con nulos en 'Std' y 'Variance'
    df = df[df['Std'].notna() & df['Variance'].notna()]
    print("Filas con valores nulos eliminadas.")
    
    # D. Eliminar registros duplicados
    df.drop_duplicates(inplace=True)
    print(f"Registros restantes tras la limpieza: {len(df):,}")
    
    # 3. Mapeo de clases (Paso 4)
    print("\n--- Paso 4: Mapeo de Clases ---")
    clases_a_categorias = {
        'DDOS-ACK_FRAGMENTATION': 'DDoS', 'DDOS-HTTP_FLOOD': 'DDoS', 'DDOS-ICMP_FLOOD': 'DDoS',
        'DDOS-ICMP_FRAGMENTATION': 'DDoS', 'DDOS-PSHACK_FLOOD': 'DDoS', 'DDOS-RSTFINFLOOD': 'DDoS',
        'DDOS-SLOWLORIS': 'DDoS', 'DDOS-SYN_FLOOD': 'DDoS', 'DDOS-SYNONYMOUSIP_FLOOD': 'DDoS',
        'DDOS-TCP_FLOOD': 'DDoS', 'DDOS-UDP_FLOOD': 'DDoS', 'DDOS-UDP_FRAGMENTATION': 'DDoS',
        'DOS-HTTP_FLOOD': 'DoS', 'DOS-SYN_FLOOD': 'DoS', 'DOS-TCP_FLOOD': 'DoS', 'DOS-UDP_FLOOD': 'DoS',
        'MIRAI-GREETH_FLOOD': 'Mirai', 'MIRAI-GREIP_FLOOD': 'Mirai', 'MIRAI-UDPPLAIN': 'Mirai',
        'RECON-HOSTDISCOVERY': 'Recon', 'RECON-OSSCAN': 'Recon', 'RECON-PINGSWEEP': 'Recon',
        'RECON-PORTSCAN': 'Recon', 'VULNERABILITYSCAN': 'Recon',
        'MITM-ARPSPOOFING': 'Spoofing', 'DNS_SPOOFING': 'Spoofing',
        'DICTIONARYBRUTEFORCE': 'Brute Force',
        'BROWSERHIJACKING': 'Web-based', 'COMMANDINJECTION': 'Web-based', 'SQLINJECTION': 'Web-based',
        'XSS': 'Web-based', 'BACKDOOR_MALWARE': 'Web-based', 'UPLOADING_ATTACK': 'Web-based',
        'BENIGN': 'Benign'
    }
    df['Label'] = df['Label'].map(clases_a_categorias)
    
    X = df.drop(columns=['Label'])
    y = df['Label']
    
    # Codificación de etiquetas
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    label_mapping = dict(zip(le.classes_, map(int, range(len(le.classes_)))))
    os.makedirs("results", exist_ok=True)
    with open("results/label_mapping.json", "w") as f:
        json.dump(label_mapping, f, indent=4)
        
    # 4. Partición 70/15/15 (Paso 5)
    print("\n--- Paso 5: Partición de Datos (70/15/15) ---")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"Entrenamiento: {X_train.shape[0]:,} muestras")
    print(f"Validación: {X_val.shape[0]:,} muestras")
    print(f"Prueba: {X_test.shape[0]:,} muestras")
    
    # 5. Selección de características (Paso 6)
    print("\n--- Paso 6: Selección de Características ---")
    _, X_sub, _, y_sub = train_test_split(
        X_train, y_train, test_size=0.05, random_state=42, stratify=y_train
    )
    print(f"Entrenando Random Forest en muestra de {X_sub.shape[0]:,} para obtener importancias...")
    rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    rf.fit(X_sub, y_sub)
    
    importances = rf.feature_importances_
    df_imp = pd.DataFrame({'Feature': X_train.columns, 'Importance': importances}).sort_values(by='Importance', ascending=False)
    
    # Graficar importancia
    plt.figure(figsize=(10, 8))
    sns.barplot(data=df_imp.head(20), x='Importance', y='Feature', palette='viridis', hue='Feature', legend=False)
    plt.title("Top 20 Características por Importancia (Random Forest)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("results/figures/14_seleccion_caracteristicas.png", dpi=300)
    plt.close()
    
    umbral = 0.001
    selected_features = df_imp[df_imp['Importance'] >= umbral]['Feature'].tolist()
    print(f"Características seleccionadas (importancia >= {umbral}): {len(selected_features)} de {X_train.shape[1]}")
    
    with open("results/selected_features.json", "w") as f:
        json.dump(selected_features, f)
        
    X_train = X_train[selected_features]
    X_val = X_val[selected_features]
    X_test = X_test[selected_features]
    
    # 6. Escalado con RobustScaler (Paso 7)
    print("\n--- Paso 7: Escalado con RobustScaler ---")
    scaler = RobustScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=selected_features, index=X_train.index)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=selected_features, index=X_val.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=selected_features, index=X_test.index)
    
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")
    
    X_val_scaled.to_csv("data/processed/X_val.csv", index=False)
    pd.DataFrame(y_val, columns=['Label']).to_csv("data/processed/y_val.csv", index=False)
    X_test_scaled.to_csv("data/processed/X_test.csv", index=False)
    pd.DataFrame(y_test, columns=['Label']).to_csv("data/processed/y_test.csv", index=False)
    
    # 7. Borderline-SMOTE (Paso 8)
    print("\n--- Paso 8: Borderline-SMOTE (Solo entrenamiento) ---")
    class_names = le.classes_
    original_counts = pd.Series(y_train).value_counts().sort_index()
    
    web_idx = label_mapping['Web-based']
    brute_idx = label_mapping['Brute Force']
    
    borderline_strategy = {
        web_idx: 100000,
        brute_idx: 100000
    }
    
    bsmote = BorderlineSMOTE(sampling_strategy=borderline_strategy, random_state=42, kind='borderline-1')
    X_train_bal, y_train_bal = bsmote.fit_resample(X_train_scaled, y_train)
    
    balanced_counts = pd.Series(y_train_bal).value_counts().sort_index()
    print("\nDistribución final en Entrenamiento tras Borderline-SMOTE:")
    for idx, count in balanced_counts.items():
        print(f"  - {class_names[idx]} (clase {idx}): {count:,}")
        
    X_train_bal.to_csv("data/balanced/X_train_borderline.csv", index=False)
    pd.DataFrame(y_train_bal, columns=['Label']).to_csv("data/balanced/y_train_borderline.csv", index=False)
    print("Datos de entrenamiento balanceados guardados en data/balanced/X_train_borderline.csv")
    
    # Graficar
    os.makedirs("results/figures", exist_ok=True)
    df_plot = pd.DataFrame({
        'Clase': [class_names[i] for i in range(len(class_names))] * 2,
        'Cantidad': list(original_counts) + list(balanced_counts),
        'Estado': ['Original'] * len(class_names) + ['Con Borderline-SMOTE'] * len(class_names)
    })
    
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_plot, x='Clase', y='Cantidad', hue='Estado', palette=['#E74C3C', '#2ECC71'])
    plt.title("Distribución del Dataset de Entrenamiento: Original vs. Borderline-SMOTE", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Categoría de Tráfico")
    plt.ylabel("Cantidad de Registros")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("results/figures/15_distribucion_borderline_smote.png", dpi=300)
    plt.close()
    
    print("\n¡Pipeline de preprocesamiento y Borderline-SMOTE completado exitosamente!")

if __name__ == "__main__":
    ejecutar_preprocesamiento()
