import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def ejecutar_eda_visual():
    print("Iniciando ejecución de EDA Visual...")
    
    # 1. Configuración de estilo
    sns.set_theme(style="whitegrid")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    
    os.makedirs("results/figures", exist_ok=True)
    
    # 2. Cargar datos
    df_path = "data/processed/dataset_consolidado.csv"
    print("Cargando dataset consolidado...")
    df = pd.read_csv(df_path)
    print(f"Dataset cargado. Filas: {len(df):,}")
    
    # 3. Distribución de clases
    print("Generando gráfico de distribución de clases...")
    class_counts = df['Label'].value_counts()
    plt.figure(figsize=(12, 10))
    colors = sns.color_palette("viridis", len(class_counts))
    sns.barplot(y=class_counts.index, x=class_counts.values, palette=colors, hue=class_counts.index, legend=False)
    plt.title("Distribución de Clases en el Dataset Consolidado (CICIoT2023)", fontsize=16, fontweight='bold', pad=15)
    plt.xlabel("Número de Registros", fontsize=12)
    plt.ylabel("Etiqueta de Tráfico / Ataque", fontsize=12)
    for index, value in enumerate(class_counts.values):
        plt.text(value + (max(class_counts.values)*0.005), index, f"{value:,}", va='center', fontsize=9)
    plt.tight_layout()
    plt.savefig("results/figures/01_distribucion_clases.png", dpi=300)
    plt.close()
    
    # 4. Matriz de correlación
    print("Generando matriz de correlación...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df_numeric = df[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    features_interes = [
        'Header_Length', 'Protocol Type', 'Time_To_Live', 'Rate', 'ack_count', 
        'syn_count', 'HTTP', 'HTTPS', 'Tot sum', 'Min', 'Max', 'AVG', 
        'Std', 'Tot size', 'Number'
    ]
    selected_features = [f for f in features_interes if f in df_numeric.columns]
    corr_matrix = df_numeric[selected_features].corr()
    plt.figure(figsize=(13, 11))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5, square=True, 
                cbar_kws={"shrink": .8}, annot_kws={"size": 9})
    plt.title("Matriz de Correlación de Pearson (Top 15 Características Clave)", fontsize=16, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig("results/figures/02_matriz_correlacion.png", dpi=300)
    plt.close()
    
    # 5. Boxplots
    print("Generando boxplots...")
    boxplot_features = ['Number', 'Time_To_Live', 'Rate', 'Std', 'Tot size', 'Header_Length']
    boxplot_features = [f for f in boxplot_features if f in df.columns]
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    for i, col in enumerate(boxplot_features):
        data_clean = df[col].replace([np.inf, -np.inf], np.nan).dropna()
        sns.boxplot(y=data_clean, ax=axes[i], color=sns.color_palette("Blues")[4])
        axes[i].set_title(f"Distribución y Outliers de '{col}'", fontsize=12, fontweight='bold')
        axes[i].set_ylabel("Valores")
        axes[i].set_xlabel("")
    plt.tight_layout()
    plt.savefig("results/figures/03_boxplots_outliers.png", dpi=300)
    plt.close()
    
    # 6. Histogramas
    print("Generando histogramas...")
    hist_features = ['Number', 'Time_To_Live', 'HTTPS', 'Std', 'Rate', 'Tot size']
    hist_features = [f for f in hist_features if f in df.columns]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for i, col in enumerate(hist_features):
        data_clean = df[col].replace([np.inf, -np.inf], np.nan).dropna()
        log_scale = False
        if data_clean.max() > 1000 and col != 'Time_To_Live':
            log_scale = True
            if (data_clean <= 0).any():
                data_clean = data_clean + 1e-5
        sns.histplot(data_clean, bins=30, ax=axes[i], kde=True, color="#4A90E2", log_scale=log_scale)
        title_str = f"Distribución de {col}" + (" (Escala Log)" if log_scale else "")
        axes[i].set_title(title_str, fontsize=12, fontweight='bold')
        axes[i].set_ylabel("Frecuencia")
        axes[i].set_xlabel("Valor")
    plt.tight_layout()
    plt.savefig("results/figures/04_histogramas.png", dpi=300)
    plt.close()
    
    print("¡EDA Visual completado! Gráficos guardados en 'results/figures/'")

if __name__ == "__main__":
    ejecutar_eda_visual()
