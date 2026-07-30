import os
import sys

def verify_environment():
    print("=== VERIFICACIÓN DEL ENTORNO DE DESARROLLO ===")
    print(f"Versión de Python: {sys.version}")
    print(f"Ruta del ejecutable Python: {sys.executable}")
    
    print("\n--- Librerías Instaladas ---")
    
    libs = ['pandas', 'numpy', 'scikit-learn', 'xgboost', 'lightgbm', 'imblearn', 'torch']
    for lib in libs:
        try:
            if lib == 'scikit-learn':
                import sklearn
                print(f"Scikit-learn: {sklearn.__version__}")
            elif lib == 'imblearn':
                import imblearn
                print(f"Imbalanced-learn (SMOTE): {imblearn.__version__}")
            else:
                imported = __import__(lib)
                print(f"{lib.capitalize()}: {imported.__version__}")
        except ImportError:
            print(f"{lib.capitalize()}: NO INSTALADO")
        
    print("\n--- Soporte de Hardware / PyTorch ---")
    try:
        import torch
        print(f"Versión de PyTorch: {torch.__version__}")
        cuda_available = torch.cuda.is_available()
        print(f"¿CUDA disponible?: {cuda_available}")
        if cuda_available:
            print(f"  Dispositivo CUDA actual: {torch.cuda.current_device()}")
            print(f"  Nombre del dispositivo: {torch.cuda.get_device_name(0)}")
            print(f"  Versión de CUDA (PyTorch): {torch.version.cuda}")
    except ImportError:
        print("PyTorch: NO INSTALADO")
        
    print("\n--- Verificación de Datos de Entrada (CICIoT2023) ---")
    raw_dir = "data/raw"
    if os.path.exists(raw_dir):
        files = [f for f in os.listdir(raw_dir) if f.endswith(".csv")]
        print(f"Directorio '{raw_dir}' encontrado.")
        print(f"Cantidad de archivos CSV encontrados: {len(files)}")
        if len(files) > 0:
            print("Primeros 5 archivos:")
            for f in sorted(files)[:5]:
                print(f"  - {f}")
            # Intento de cargar las primeras líneas de Merged01.csv
            m01_path = os.path.join(raw_dir, "Merged01.csv")
            try:
                import pandas as pd
                df_head = pd.read_csv(m01_path, nrows=5)
                print(f"\nLectura exitosa de {m01_path}:")
                print(f"  Columnas: {list(df_head.columns[:5])}... (Total: {len(df_head.columns)} columnas)")
                print(f"  Dimensiones de muestra: {df_head.shape}")
            except Exception as e:
                print(f"  Error al leer {m01_path}: {e}")
        else:
            print("  [ERROR] No se encontraron archivos CSV en data/raw/")
    else:
        print(f"  [ERROR] El directorio '{raw_dir}' no existe.")

if __name__ == "__main__":
    verify_environment()
