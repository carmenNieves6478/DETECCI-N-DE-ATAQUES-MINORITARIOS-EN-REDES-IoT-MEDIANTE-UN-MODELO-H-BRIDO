import os
import pandas as pd
from tqdm import tqdm

# Mapping dictionary for the 33 attacks in CICIoT2023 to 8 high-level categories
LABEL_MAP = {
    # DDoS (12 attacks)
    'DDOS-ICMP_FLOOD': 'DDoS',
    'DDOS-UDP_FLOOD': 'DDoS',
    'DDOS-TCP_FLOOD': 'DDoS',
    'DDOS-PSHACK_FLOOD': 'DDoS',
    'DDOS-RSTFINFLOOD': 'DDoS',
    'DDOS-SYN_FLOOD': 'DDoS',
    'DDOS-SYNONYMOUSIP_FLOOD': 'DDoS',
    'DDOS-ICMP_FRAGMENTATION': 'DDoS',
    'DDOS-ACK_FRAGMENTATION': 'DDoS',
    'DDOS-UDP_FRAGMENTATION': 'DDoS',
    'DDOS-HTTP_FLOOD': 'DDoS',
    'DDOS-SLOWLORIS': 'DDoS',
    
    # DoS (4 attacks)
    'DOS-UDP_FLOOD': 'DoS',
    'DOS-TCP_FLOOD': 'DoS',
    'DOS-SYN_FLOOD': 'DoS',
    'DOS-HTTP_FLOOD': 'DoS',
    
    # Mirai (5 attacks/variants)
    'MIRAI-GREETH_FLOOD': 'Mirai',
    'MIRAI-UDPPLAIN': 'Mirai',
    'MIRAI-GREIP_FLOOD': 'Mirai',
    'MIRAI-HTTP_FLOOD': 'Mirai',
    'MIRAI-ACK_FLOOD': 'Mirai',
    
    # Spoofing (2 attacks)
    'MITM-ARPSPOOFING': 'Spoofing',
    'DNS_SPOOFING': 'Spoofing',
    
    # Reconnaissance / Scanning (5 attacks)
    'VULNERABILITYSCAN': 'Recon',
    'RECON-HOSTDISCOVERY': 'Recon',
    'RECON-OSSCAN': 'Recon',
    'RECON-PORTSCAN': 'Recon',
    'RECON-PINGSWEEP': 'Recon',
    
    # Web-based / Malware (6 attacks)
    'BROWSERHIJACKING': 'Web-based',
    'SQLINJECTION': 'Web-based',
    'COMMANDINJECTION': 'Web-based',
    'XSS': 'Web-based',
    'UPLOADING_ATTACK': 'Web-based',
    'BACKDOOR_MALWARE': 'Web-based',
    
    # Brute Force (1 attack)
    'DICTIONARYBRUTEFORCE': 'Brute Force',
    
    # Benign traffic (1 class)
    'BENIGN': 'Benign',
    'BENIGNTRAFFIC': 'Benign'
}

def process_and_sample_dataset(raw_dir='data/raw', output_path='data/processed/sampled_dataset.parquet'):
    """
    Slices and samples the CICIoT2023 dataset incrementally file-by-file to solve
    memory overhead and class imbalance. Saves the output to a Parquet file.
    """
    # Define sampling rates per category
    # Downsamples majority classes and preserves 100% of rare minority classes
    rates = {
        'DDoS': 0.05,        # 5% (from ~29M to ~1.45M)
        'DoS': 0.10,         # 10% (from ~9M to ~900k)
        'Mirai': 0.10,       # 10% (from ~5M to ~500k)
        'Benign': 0.50,      # 50% (from ~1M to ~500k)
        'Recon': 0.50,       # 50% (from ~800k to ~400k)
        'Spoofing': 1.00,    # 100% (preserves all ~300k)
        'Brute Force': 1.00, # 100% (preserves all ~40k)
        'Web-based': 1.00    # 100% (preserves all ~15k)
    }
    
    # Verify the raw directory exists (which is the symlink data/raw)
    if not os.path.exists(raw_dir):
        raise FileNotFoundError(f"El directorio raw '{raw_dir}' no existe. Asegúrate de haber completado el Paso 0.")
        
    csv_files = [f for f in os.listdir(raw_dir) if f.endswith('.csv') and f.startswith('Merged')]
    csv_files.sort()
    
    if not csv_files:
        raise FileNotFoundError(f"No se encontraron archivos CSV que comiencen con 'Merged' en {raw_dir}")
        
    print(f"Se encontraron {len(csv_files)} archivos CSV para procesar.")
    sampled_chunks = []
    
    for file_name in tqdm(csv_files, desc="Procesando archivos"):
        file_path = os.path.join(raw_dir, file_name)
        
        # Load the CSV
        df = pd.read_csv(file_path)
        
        # Normalize and map labels
        df['Category'] = df['Label'].str.strip().str.upper().map(LABEL_MAP).fillna('Unknown')
        
        # Check for unmapped labels
        unknowns = df[df['Category'] == 'Unknown']['Label'].unique()
        if len(unknowns) > 0:
            print(f"\nAdvertencia en {file_name}: Etiquetas desconocidas encontradas: {unknowns}")
            
        sampled_dfs = []
        for category, group in df.groupby('Category'):
            rate = rates.get(category, 1.0)
            
            # Keep 100% of small classes or groups with fewer than 1000 records to avoid data loss
            if rate >= 1.0 or len(group) < 1000:
                sampled_dfs.append(group)
            else:
                sampled = group.sample(frac=rate, random_state=42)
                # Safety check to avoid empty groups
                if len(sampled) == 0 and len(group) > 0:
                    sampled = group.sample(n=1, random_state=42)
                sampled_dfs.append(sampled)
                
        sampled_file_df = pd.concat(sampled_dfs, ignore_index=True)
        sampled_chunks.append(sampled_file_df)
        
    # Concatenate all sampled chunks
    print("\nConcatenando muestras...")
    final_df = pd.concat(sampled_chunks, ignore_index=True)
    
    # Save as Parquet
    print(f"Guardando dataset muestreado en {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_df.to_parquet(output_path, index=False)
    
    print("\n¡Proceso completado exitosamente!")
    print(f"Dimensiones del dataset final: {final_df.shape}")
    print("\nDistribución final de categorías:")
    counts = final_df['Category'].value_counts()
    percentages = final_df['Category'].value_counts(normalize=True) * 100
    for cat in counts.index:
        print(f" - {cat}: {counts[cat]} registros ({percentages[cat]:.2f}%)")
        
    return final_df

if __name__ == "__main__":
    process_and_sample_dataset()
