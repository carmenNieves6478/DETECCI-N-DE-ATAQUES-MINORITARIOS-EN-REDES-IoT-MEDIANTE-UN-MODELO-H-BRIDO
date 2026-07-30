import os
import pandas as pd
import numpy as np

def limpiar_datos():
    input_path = "data/processed/dataset_consolidado.csv"
    output_path = "data/processed/dataset_limpio.csv"
    
    print("Cargando dataset consolidado...")
    df = pd.read_csv(input_path)
    initial_rows = len(df)
    
    print(f"Registros iniciales: {initial_rows:,}")
    
    # 1. Manejo de valores duplicados
    print("\n--- 1. Eliminando filas exactamente duplicadas ---")
    df.drop_duplicates(inplace=True)
    rows_after_dup = len(df)
    dropped_dup = initial_rows - rows_after_dup
    print(f"Filas duplicadas eliminadas: {dropped_dup:,} ({dropped_dup/initial_rows*100:.2f}%)")
    print(f"Registros restantes: {rows_after_dup:,}")
    
    # 2. Manejo de valores nulos (NaN)
    print("\n--- 2. Tratamiento de valores nulos (NaN) ---")
    # Imputar Std y Variance con 0 (ya que flujos de un solo paquete tienen std/variance de 0)
    cols_to_fill_0 = ['Std', 'Variance']
    for col in cols_to_fill_0:
        if col in df.columns:
            nan_count = df[col].isna().sum()
            df[col] = df[col].fillna(0.0)
            print(f"Columna '{col}': se imputaron {nan_count} nulos con 0.0")
            
    # 3. Manejo de valores infinitos (inf)
    print("\n--- 3. Tratamiento de valores infinitos (inf) ---")
    if 'Rate' in df.columns:
        # Reemplazar valores infinitos por NaN para identificar el máximo valor finito
        df['Rate'] = df['Rate'].replace([np.inf, -np.inf], np.nan)
        max_rate = df['Rate'].max()
        print(f"Valor máximo finito en 'Rate': {max_rate}")
        
        # Imputar los infinitos con el valor máximo finito encontrado
        df['Rate'] = df['Rate'].fillna(max_rate)
        print("Valores infinitos en 'Rate' imputados con el máximo valor finito.")
        
    # Verificar que no queden nulos ni infinitos en el dataset
    nan_remaining = df.isna().sum().sum()
    inf_remaining = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
    print(f"\nVerificación final:")
    print(f"  Nulos restantes: {nan_remaining}")
    print(f"  Infinitos restantes: {inf_remaining}")
    
    # 4. Mostrar y guardar la distribución de clases limpia
    print("\n--- 4. Distribución de clases en el dataset limpio ---")
    class_counts = df['Label'].value_counts()
    dist_limpia = pd.DataFrame({
        'Cantidad': class_counts,
        'Porcentaje': (class_counts / len(df) * 100).round(4)
    })
    print(dist_limpia)
    
    # Guardar en data/processed/
    print(f"\nGuardando dataset limpio en: {output_path}...")
    df.to_csv(output_path, index=False)
    print("¡Guardado completado exitosamente!")
    
    # Guardar distribución para documentación
    os.makedirs("results", exist_ok=True)
    dist_limpia.to_csv("results/distribucion_clases_limpio.csv")

if __name__ == "__main__":
    limpiar_datos()
