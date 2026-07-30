import os
import pandas as pd
import numpy as np
from tqdm import tqdm

def check_raw_data():
    raw_dir = "data/raw"
    files = sorted([f for f in os.listdir(raw_dir) if f.endswith(".csv")])
    
    print(f"Total de archivos CSV a procesar: {len(files)}")
    
    # 1. Obtener información de columnas del primer archivo
    first_file = os.path.join(raw_dir, files[0])
    df_sample = pd.read_csv(first_file, nrows=5)
    
    print("\n--- Columnas del Dataset ---")
    print(list(df_sample.columns))
    
    # Comprobar el nombre de la columna objetivo (normalmente 'label')
    target_col = 'label' if 'label' in df_sample.columns else None
    if target_col is None:
        # Buscar columnas que parezcan etiquetas
        label_cols = [c for c in df_sample.columns if 'label' in c.lower() or 'class' in c.lower()]
        if label_cols:
            target_col = label_cols[0]
            print(f"Columna objetivo detectada: '{target_col}'")
        else:
            print("No se pudo detectar automáticamente la columna objetivo.")
            return
    else:
        print(f"Columna objetivo: '{target_col}'")
        
    # 2. Análisis rápido de tamaño y distribución de clases en una muestra de archivos
    print("\n--- Perfilado rápido de clases (Archivos 1 a 5) ---")
    class_counts = pd.Series(dtype=int)
    total_rows_sample = 0
    
    # Leemos solo los primeros 5 archivos para no tardar mucho en esta verificación
    for f in tqdm(files[:5], desc="Procesando muestra"):
        file_path = os.path.join(raw_dir, f)
        df = pd.read_csv(file_path, usecols=[target_col])
        class_counts = class_counts.add(df[target_col].value_counts(), fill_value=0)
        total_rows_sample += len(df)
        
    print(f"\nTotal de filas en los primeros 5 archivos: {total_rows_sample:,}")
    print("\nDistribución de clases en los primeros 5 archivos:")
    dist = pd.DataFrame({
        'Cantidad': class_counts.astype(int),
        'Porcentaje': (class_counts / total_rows_sample * 100).round(4)
    }).sort_values(by='Cantidad', ascending=False)
    print(dist)
    
    # Guardar este perfil temporal en results/
    os.makedirs("results", exist_ok=True)
    dist.to_csv("results/distribucion_clases_muestra.csv")
    print("\nDistribución guardada en 'results/distribucion_clases_muestra.csv'")

if __name__ == "__main__":
    check_raw_data()
