import os
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

def ejecutar_preparacion_ml():
    print("Iniciando Pipeline de Preparación para ML y SMOTE (Paso 5)...")
    
    # 1. Cargar datos mapeados
    df_path = "data/processed/dataset_mapeado.csv"
    df = pd.read_csv(df_path)
    print(f"Dataset mapeado cargado. Filas: {len(df):,}")
    
    # 2. Codificación con LabelEncoder
    le = LabelEncoder()
    df['Label_encoded'] = le.fit_transform(df['Label'])
    
    label_mapping = dict(zip(le.classes_, map(int, range(len(le.classes_)))))
    print("\nMapeo de categorías a enteros:")
    for k, v in label_mapping.items():
        print(f"  - {k} -> {v}")
        
    os.makedirs("results", exist_ok=True)
    with open("results/label_mapping.json", "w") as f:
        json.dump(label_mapping, f, indent=4)
    print("Mapeo guardado en 'results/label_mapping.json'")
    
    df.drop(columns=['Label'], inplace=True)
    df.rename(columns={'Label_encoded': 'Label'}, inplace=True)
    
    # 3. Separar X e y y hacer Split estratificado
    X = df.drop(columns=['Label'])
    y = df['Label']
    
    print("\nRealizando partición de datos Train/Test estratificada (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Conjunto de Entrenamiento: {X_train.shape[0]:,} muestras")
    print(f"Conjunto de Prueba: {X_test.shape[0]:,} muestras")
    
    # Guardar conjunto de prueba inmediatamente
    print("Guardando conjunto de prueba...")
    X_test.to_csv("data/processed/X_test.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)
    print("¡Conjunto de prueba guardado!")
    
    # 4. Aplicar SMOTE únicamente al conjunto de entrenamiento
    class_names = le.classes_
    original_train_counts = y_train.value_counts().sort_index()
    print("\nDistribución original en Entrenamiento:")
    for idx, count in original_train_counts.items():
        print(f"  - {class_names[idx]} (clase {idx}): {count:,}")
        
    web_idx = label_mapping['Web-based']
    brute_idx = label_mapping['Brute Force']
    
    smote_strategy = {
        web_idx: 100000,
        brute_idx: 100000
    }
    
    print(f"\nAplicando SMOTE con estrategia: {smote_strategy}...")
    smote = SMOTE(sampling_strategy=smote_strategy, random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    resampled_train_counts = y_train_res.value_counts().sort_index()
    print("\nDistribución final en Entrenamiento tras SMOTE:")
    for idx, count in resampled_train_counts.items():
        print(f"  - {class_names[idx]} (clase {idx}): {count:,}")
        
    # Guardar conjunto de entrenamiento balanceado
    print("\nGuardando conjunto de entrenamiento balanceado...")
    X_train_res.to_csv("data/processed/X_train_resampled.csv", index=False)
    y_train_res.to_csv("data/processed/y_train_resampled.csv", index=False)
    print("¡Conjunto de entrenamiento balanceado guardado!")
    
    # 5. Generar y guardar gráfico comparativo
    print("\nGenerando gráfico de distribución con SMOTE...")
    indices = range(len(class_names))
    labels = [class_names[i] for i in indices]
    
    df_plot = pd.DataFrame({
        'Clase': labels * 2,
        'Cantidad': list(original_train_counts) + list(resampled_train_counts),
        'Estado': ['Original'] * len(labels) + ['Con SMOTE'] * len(labels)
    })
    
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_plot, x='Clase', y='Cantidad', hue='Estado', palette=['#E74C3C', '#2ECC71'])
    plt.title("Comparación de la Distribución de Clases en Entrenamiento (Original vs. SMOTE)", 
              fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Categoría de Tráfico", fontsize=11)
    plt.ylabel("Cantidad de Registros", fontsize=11)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("results/figures/07_distribucion_smote.png", dpi=300)
    plt.close()
    print("Gráfico guardado en: results/figures/07_distribucion_smote.png")
    
    print("\n¡Pipeline de preparación para ML y SMOTE completado con éxito!")

if __name__ == "__main__":
    ejecutar_preparacion_ml()
