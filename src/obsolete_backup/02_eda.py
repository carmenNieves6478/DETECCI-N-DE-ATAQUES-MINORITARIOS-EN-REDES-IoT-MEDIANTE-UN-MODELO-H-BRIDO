import os
import pandas as pd
import numpy as np

def realizar_eda():
    consolidated_path = "data/processed/dataset_consolidado.csv"
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    print("Cargando dataset consolidado para análisis...")
    df = pd.read_csv(consolidated_path)
    
    print("\n--- 1. Dimensiones del Dataset ---")
    print(f"Filas: {df.shape[0]:,}")
    print(f"Columnas: {df.shape[1]}")
    
    print("\n--- 2. Tipos de Datos y Valores Nulos/Infinitos ---")
    # Buscar tipos de datos
    dtypes_count = df.dtypes.value_counts()
    print("Tipos de columnas detectados:")
    for dtype, count in dtypes_count.items():
        print(f"  - {dtype}: {count} columnas")
        
    # Verificar columnas no numéricas (excepto Label)
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if 'Label' in non_numeric_cols:
        non_numeric_cols.remove('Label')
    print(f"Columnas no numéricas adicionales: {non_numeric_cols}")
    
    # Verificar valores nulos (NaN)
    nan_counts = df.isna().sum()
    total_nans = nan_counts.sum()
    print(f"Total de valores nulos (NaN): {total_nans}")
    if total_nans > 0:
        print("Columnas con valores nulos:")
        print(nan_counts[nan_counts > 0])
        
    # Verificar valores infinitos (inf, -inf)
    # Solo aplicable a columnas numéricas
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_counts = np.isinf(df[numeric_cols]).sum()
    total_infs = inf_counts.sum()
    print(f"Total de valores infinitos (inf/-inf): {total_infs}")
    if total_infs > 0:
        print("Columnas con valores infinitos:")
        print(inf_counts[inf_counts > 0])
        
    print("\n--- 3. Verificación de Filas Duplicadas ---")
    # Nota: En ciberseguridad, flujos idénticos pueden ocurrir, pero es bueno saber cuántos hay
    duplicate_count = df.duplicated().sum()
    print(f"Filas exactamente duplicadas: {duplicate_count:,} ({duplicate_count/len(df)*100:.2f}%)")
    
    print("\n--- 4. Estadísticas Descriptivas Básicas ---")
    # Calcular estadísticas descriptivas y guardarlas
    stats = df.describe().transpose()
    stats_output = os.path.join(results_dir, "eda_estadisticas_descriptivas.csv")
    stats.to_csv(stats_output)
    print(f"Estadísticas descriptivas guardadas en: {stats_output}")
    
    # Mostrar resumen de las columnas principales
    print("\nMuestra de estadísticas descriptivas (primeras 5 columnas):")
    print(stats.head(5))
    
    print("\n¡EDA preliminar finalizado con éxito!")

if __name__ == "__main__":
    realizar_eda()
