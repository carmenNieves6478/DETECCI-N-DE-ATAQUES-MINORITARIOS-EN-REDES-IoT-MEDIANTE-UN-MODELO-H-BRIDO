import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def ejecutar_mapeo():
    print("Iniciando Pipeline de Mapeo de Clases (Paso 4)...")
    
    # 1. Cargar datos limpios
    df_path = "data/processed/dataset_limpio.csv"
    df = pd.read_csv(df_path)
    print(f"Dataset limpio cargado. Filas: {len(df):,}")
    
    # 2. Definición del diccionario de mapeo
    clases_a_categorias = {
        # DDoS
        'DDOS-ACK_FRAGMENTATION': 'DDoS',
        'DDOS-HTTP_FLOOD': 'DDoS',
        'DDOS-ICMP_FLOOD': 'DDoS',
        'DDOS-ICMP_FRAGMENTATION': 'DDoS',
        'DDOS-PSHACK_FLOOD': 'DDoS',
        'DDOS-RSTFINFLOOD': 'DDoS',
        'DDOS-SLOWLORIS': 'DDoS',
        'DDOS-SYN_FLOOD': 'DDoS',
        'DDOS-SYNONYMOUSIP_FLOOD': 'DDoS',
        'DDOS-TCP_FLOOD': 'DDoS',
        'DDOS-UDP_FLOOD': 'DDoS',
        'DDOS-UDP_FRAGMENTATION': 'DDoS',
        
        # DoS
        'DOS-HTTP_FLOOD': 'DoS',
        'DOS-SYN_FLOOD': 'DoS',
        'DOS-TCP_FLOOD': 'DoS',
        'DOS-UDP_FLOOD': 'DoS',
        
        # Mirai
        'MIRAI-GREETH_FLOOD': 'Mirai',
        'MIRAI-GREIP_FLOOD': 'Mirai',
        'MIRAI-UDPPLAIN': 'Mirai',
        
        # Recon
        'RECON-HOSTDISCOVERY': 'Recon',
        'RECON-OSSCAN': 'Recon',
        'RECON-PINGSWEEP': 'Recon',
        'RECON-PORTSCAN': 'Recon',
        'VULNERABILITYSCAN': 'Recon',
        
        # Spoofing
        'MITM-ARPSPOOFING': 'Spoofing',
        'DNS_SPOOFING': 'Spoofing',
        
        # Brute Force
        'DICTIONARYBRUTEFORCE': 'Brute Force',
        
        # Web-based
        'BROWSERHIJACKING': 'Web-based',
        'COMMANDINJECTION': 'Web-based',
        'SQLINJECTION': 'Web-based',
        'XSS': 'Web-based',
        'BACKDOOR_MALWARE': 'Web-based',
        'UPLOADING_ATTACK': 'Web-based',
        
        # Benign
        'BENIGN': 'Benign'
    }
    
    # 3. Aplicar mapeo
    df['Label'] = df['Label'].map(clases_a_categorias)
    
    # 4. Distribución de categorías
    class_counts = df['Label'].value_counts()
    dist_final = pd.DataFrame({
        'Cantidad': class_counts,
        'Porcentaje': (class_counts / len(df) * 100).round(4)
    })
    
    os.makedirs("results", exist_ok=True)
    dist_final.to_csv("results/distribucion_categorias.csv")
    print("\nDistribución de las 8 categorías:")
    print(dist_final)
    
    # 5. Generar gráfico
    print("\nGenerando gráfico de distribución de categorías...")
    os.makedirs("results/figures", exist_ok=True)
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    colors = sns.color_palette("coolwarm", len(class_counts))
    sns.barplot(y=class_counts.index, x=class_counts.values, palette=colors, hue=class_counts.index, legend=False)
    plt.title("Distribución de las 8 Categorías de Tráfico (CICIoT2023)", fontsize=15, fontweight='bold', pad=15)
    plt.xlabel("Cantidad de Registros", fontsize=11)
    plt.ylabel("Categoría", fontsize=11)
    
    for index, value in enumerate(class_counts.values):
        plt.text(value + (max(class_counts.values)*0.005), index, f"{value:,} ({value/len(df)*100:.2f}%)", va='center', fontsize=9)
        
    plt.tight_layout()
    plt.savefig("results/figures/06_distribucion_categorias.png", dpi=300)
    plt.close()
    print("Gráfico guardado en: results/figures/06_distribucion_categorias.png")
    
    # 6. Guardar dataset mapeado
    output_path = "data/processed/dataset_mapeado.csv"
    df.to_csv(output_path, index=False)
    print(f"Dataset mapeado guardado en: {output_path}")
    
    print("\n¡Pipeline de mapeo completado con éxito!")

if __name__ == "__main__":
    ejecutar_mapeo()
