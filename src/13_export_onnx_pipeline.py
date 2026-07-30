import os
import joblib
import numpy as np
import pandas as pd
import time
import onnxruntime as ort
import onnxmltools
from onnxmltools.convert import convert_lightgbm, convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType

def exportar_modelos_onnx():
    print("======================================================================")
    print("PIPELINE DE EXPORTACIÓN Y DESPLIEGUE MULTI-ARQUITECTURA (ONNX - IoT)")
    print("======================================================================")
    
    os.makedirs("models/onnx", exist_ok=True)
    
    # 1. Cargar el Meta-Modelo LightGBM y Modelos Base
    print("\n1. Cargando modelos entrenados desde 'models/'...")
    meta_model = joblib.load("models/meta_lightgbm.pkl")
    xgb_model = joblib.load("models/base_xgb.pkl")
    lgb_model = joblib.load("models/base_lgb.pkl")
    
    # 2. Exportar Meta-Modelo LightGBM a ONNX
    print("\n2. Exportando Meta-Modelo LightGBM (Nivel 1) a formato ONNX...")
    initial_type_meta = [('float_input', FloatTensorType([None, 27]))]
    onnx_meta = convert_lightgbm(meta_model, initial_types=initial_type_meta, target_opset=15)
    meta_onnx_path = "models/onnx/meta_lightgbm.onnx"
    with open(meta_onnx_path, "wb") as f:
        f.write(onnx_meta.SerializeToString())
    print(f"   [Éxito] Meta-Modelo exportado en: {meta_onnx_path} ({os.path.getsize(meta_onnx_path)/1024:.2f} KB)")
    
    # 3. Exportar XGBoost a ONNX (Normalizar feature names para conversor)
    print("\n3. Exportando XGBoost Base (Nivel 0) a formato ONNX...")
    initial_type_base = [('float_input', FloatTensorType([None, 29]))]
    booster = xgb_model.get_booster()
    booster.feature_names = [f"f{i}" for i in range(29)]
    onnx_xgb = convert_xgboost(xgb_model, initial_types=initial_type_base, target_opset=15)
    xgb_onnx_path = "models/onnx/base_xgb.onnx"
    with open(xgb_onnx_path, "wb") as f:
        f.write(onnx_xgb.SerializeToString())
    print(f"   [Éxito] XGBoost exportado en: {xgb_onnx_path} ({os.path.getsize(xgb_onnx_path)/1024:.2f} KB)")

    # 4. Exportar LightGBM Base a ONNX
    print("\n4. Exportando LightGBM Base (Nivel 0) a formato ONNX...")
    onnx_lgb = convert_lightgbm(lgb_model, initial_types=initial_type_base, target_opset=15)
    lgb_onnx_path = "models/onnx/base_lgb.onnx"
    with open(lgb_onnx_path, "wb") as f:
        f.write(onnx_lgb.SerializeToString())
    print(f"   [Éxito] LightGBM exportado en: {lgb_onnx_path} ({os.path.getsize(lgb_onnx_path)/1024:.2f} KB)")
    
    # 5. Prueba de Inferencia Cross-Platform con ONNX Runtime
    print("\n5. Evaluando rendimiento de inferencia con ONNX Runtime (Multi-Arquitectura CPU)...")
    X_test_meta = pd.read_csv("data/processed/X_test_meta.csv", dtype=np.float32).values
    
    session = ort.InferenceSession(meta_onnx_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    
    # Benchmark de latencia sobre 50,000 muestras
    sub_sample = X_test_meta[:50000].astype(np.float32)
    start_time = time.time()
    _ = session.run(None, {input_name: sub_sample})
    elapsed = time.time() - start_time
    latencia_us = (elapsed / len(sub_sample)) * 1e6
    
    print(f"   - Muestras probadas con ONNX Runtime: {len(sub_sample):,}")
    print(f"   - Latencia de Inferencia Meta-Modelo ONNX (CPU): {latencia_us:.2f} μs / muestra")
    print(f"   - Tasa de Procesamiento Estimada: {1e6 / latencia_us:,.0f} paquetes/segundo")
    print("\n======================================================================")
    print("¡MODELOS PORTABLES LISTOS PARA DESPLIEGUE EN PASARELAS IOT (ARM64 / x86)!")
    print("======================================================================")

if __name__ == "__main__":
    exportar_modelos_onnx()
