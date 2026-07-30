import os
import pandas as pd
import numpy as np
from tqdm import tqdm

def consolidar_y_submuestrear(samples_per_class_per_file=2000, seed=42):
    np.random.seed(seed)
    raw_dir = "data/raw"
    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)
    
    files = sorted([f for f in os.listdir(raw_dir) if f.endswith(".csv")])
    print(f"Iniciando consolidación de {len(files)} archivos...")
    print(f"Estrategia: Máximo {samples_per_class_per_file} muestras por clase por archivo. Clases menores se conservan completas.")
    
    chunks = []
    
    for f in tqdm(files, desc="Procesando archivos"):
        file_path = os.path.join(raw_dir, f)
        
        # Leer el archivo completo
        df = pd.read_csv(file_path)
        
        # Lista para almacenar el submuestreo de este archivo
        sampled_dfs = []
        
        for label, group in df.groupby('Label'):
            if len(group) > samples_per_class_per_file:
                # Submuestreo aleatorio
                sampled_group = group.sample(n=samples_per_class_per_file, random_state=seed)
                sampled_dfs.append(sampled_group)
            else:
                # Conservar 100% de la clase minoritaria
                sampled_dfs.append(group)
                
        # Concatenar las clases procesadas de este archivo
        if sampled_dfs:
            df_file_sampled = pd.concat(sampled_dfs, axis=0)
            chunks.append(df_file_sampled)
            
    # Concatenar todos los archivos submuestreados
    print("\nConcatenando todos los fragmentos procesados...")
    df_final = pd.concat(chunks, axis=0, ignore_index=True)
    
    # Mostrar resumen del dataset resultante
    print("\n--- Resumen del Dataset Consolidado ---")
    print(f"Total de registros: {len(df_final):,}")
    print(f"Total de columnas: {df_final.shape[1]}")
    
    # Distribución de clases en el dataset consolidado
    class_counts = df_final['Label'].value_counts()
    dist_final = pd.DataFrame({
        'Cantidad': class_counts,
        'Porcentaje': (class_counts / len(df_final) * 100).round(4)
    })
    print("\nDistribución final de clases:")
    print(dist_final)
    
    # Guardar en data/processed/
    output_path = os.path.join(processed_dir, "dataset_consolidado.csv")
    print(f"\nGuardando dataset consolidado en: {output_path}...")
    df_final.to_csv(output_path, index=False)
    print("¡Guardado completado exitosamente!")
    
    # Guardar distribución para documentación
    dist_final.to_csv("results/distribucion_clases_consolidado.csv")

if __name__ == "__main__":
    consolidar_y_submuestrear(samples_per_class_per_file=2000, seed=42)
