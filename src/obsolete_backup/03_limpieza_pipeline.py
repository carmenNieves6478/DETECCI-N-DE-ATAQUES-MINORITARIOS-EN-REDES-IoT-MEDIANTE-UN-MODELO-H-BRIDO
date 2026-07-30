import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler

def ejecutar_limpieza():
    print("Iniciando Pipeline de Limpieza (Paso 3)...")
    
    # 1. Cargar datos
    df_path = "data/processed/dataset_consolidado.csv"
    df = pd.read_csv(df_path)
    initial_len = len(df)
    print(f"Filas iniciales: {initial_len:,}")
    
    # 2. Eliminar columnas redundantes (AVG, Tot sum, Min)
    cols_redundantes = ['AVG', 'Tot sum', 'Min']
    cols_a_eliminar = [c for c in cols_redundantes if c in df.columns]
    print(f"Eliminando columnas redundantes: {cols_a_eliminar}")
    df.drop(columns=cols_a_eliminar, inplace=True)
    
    # 3. Eliminar filas con valores infinitos en Rate
    if 'Rate' in df.columns:
        len_before = len(df)
        inf_mask = np.isinf(df['Rate'])
        df = df[~inf_mask]
        print(f"Filas con infinitos en 'Rate' eliminadas: {len_before - len(df):,}")
        
    # 4. Eliminar filas con valores nulos en Std y Variance
    len_before = len(df)
    null_mask = df['Std'].isna() | df['Variance'].isna()
    df = df[~null_mask]
    print(f"Filas con nulos en 'Std' o 'Variance' eliminadas: {len_before - len(df):,}")
    
    # 5. Eliminar duplicados
    len_before = len(df)
    df.drop_duplicates(inplace=True)
    print(f"Filas duplicadas eliminadas: {len_before - len(df):,}")
    print(f"Registros después de la limpieza: {len(df):,}")
    
    # 6. Guardar copia sin escalar para gráficos
    label_col = 'Label'
    X_cols = [c for c in df.columns if c != label_col]
    df_before_scale = df[X_cols].copy()
    
    # 7. Aplicar RobustScaler
    print("Aplicando RobustScaler...")
    scaler = RobustScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df[X_cols]), columns=X_cols, index=df.index)
    df_final = pd.concat([df_scaled, df[label_col]], axis=1)
    
    # 8. Guardar dataset limpio
    output_path = "data/processed/dataset_limpio.csv"
    df_final.to_csv(output_path, index=False)
    print(f"Dataset limpio guardado en: {output_path}")
    
    # 9. Generar gráficos comparativos
    print("Generando gráficos comparativos de distribución...")
    cols_a_comparar = ['Rate', 'Number', 'Std', 'Tot size']
    cols_a_comparar = [c for c in cols_a_comparar if c in df_before_scale.columns]
    
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(len(cols_a_comparar), 2, figsize=(14, 3.5 * len(cols_a_comparar)))
    
    for i, col in enumerate(cols_a_comparar):
        # Original
        sns.histplot(df_before_scale[col], bins=30, ax=axes[i, 0], color="#E67E22", kde=True, stat="density")
        axes[i, 0].set_title(f"Original: {col}", fontsize=12, fontweight='bold')
        axes[i, 0].set_xlabel("Valor")
        axes[i, 0].set_ylabel("Densidad")
        
        # Escalado
        sns.histplot(df_final[col], bins=30, ax=axes[i, 1], color="#2980B9", kde=True, stat="density")
        axes[i, 1].set_title(f"RobustScaler: {col}", fontsize=12, fontweight='bold')
        axes[i, 1].set_xlabel("Valor Escalado")
        axes[i, 1].set_ylabel("Densidad")
        
    plt.suptitle("Comparativa de Distribuciones: Antes vs Después del Procesamiento y Escalado", 
                 fontsize=15, fontweight='bold', y=0.99)
    plt.tight_layout()
    plt.savefig("results/figures/05_comparacion_limpieza.png", dpi=300)
    plt.close()
    print("Gráfico guardado en: results/figures/05_comparacion_limpieza.png")
    
    # 10. Guardar distribución de clases
    class_counts = df_final['Label'].value_counts()
    dist_final = pd.DataFrame({
        'Cantidad': class_counts,
        'Porcentaje': (class_counts / len(df_final) * 100).round(4)
    })
    dist_final.to_csv("results/distribucion_clases_limpio.csv")
    print("\nDistribución final de clases:")
    print(dist_final)
    
    print("\n¡Pipeline de limpieza completado con éxito!")

if __name__ == "__main__":
    ejecutar_limpieza()
