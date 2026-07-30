# Detección de Ataques Minoritarios en Redes IoT mediante un Modelo Híbrido basado en Stacking, XGBoost y LightGBM, Puno 2026

![Python Version](https://img.shields.io/badge/python-3.10-blue.svg)
![Framework](https://img.shields.io/badge/framework-Scikit--Learn%20%7C%20XGBoost%20%7C%20LightGBM-orange.svg)
![CUDA Acceleration](https://img.shields.io/badge/CUDA-13.2-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

Este repositorio contiene la implementación oficial en Python del sistema de detección de intrusiones en redes (NIDS) basado en una arquitectura jerárquica de ensamble **Stacking de dos niveles**. El modelo combina los algoritmos **Random Forest, XGBoost GPU y LightGBM** en el Nivel 0, integrando la cuantificación de incertidumbre probabilística mediante la **Entropía de Shannon** y el sobremuestreo sintético adaptativo en la frontera a través de **Borderline-SMOTE1** sobre el dataset estandarizado **CICIoT2023**.

---

## 🌟 Características Principales

* **Arquitectura Stacking de Dos Niveles:** Combina modelos base heterogéneos en Nivel 0 y un meta-clasificador LightGBM optimizado en Nivel 1.
* **Cuantificación de Incertidumbre Entrópica:** Incorpora la Entropía de Shannon $H(x)$ de las distribuciones probabilísticas emitidas por los modelos base dentro del vector de meta-características $Z \in \mathbb{R}^{27}$.
* **Mitigación del Sesgo por Desbalance Extremo:** Aplica el algoritmo `Borderline-SMOTE1` exclusivamente en la zona de peligro limítrofe ($S_{\mathrm{danger}}$) para elevar la sensibilidad (*Recall*) en ataques infrecuentes (*Web-based* y *Brute Force*).
* **Preprocesamiento Robusto:** Normalización con `RobustScaler` basada en la Mediana y el Rango Intercuartílico ($\mathrm{IQR}$) resistente a ráfagas de valores atípicos (*outliers*).
* **Alto Rendimiento y Baja Latencia:** Diseñado para pasarelas de borde IoT con latencia de inferencia de **$8.65\,\mu s$ por flujo** y ocupación de memoria RAM de **$257.89\text{ MB}$**.
* **Validación Estadística:** Comprobación de significancia mediante la Prueba No Paramétrica de Rangos con Signo de Wilcoxon ($p = 0.003744 < 0.05$).

---

## 📁 Estructura del Repositorio

```text
.
├── main.docx                    # Documento máster de Tesis completo en formato Microsoft Word
├── presentacion_tesis.pdf       # Diapositivas Beamer en PDF para la sustentación
├── presentacion_tesis.tex       # Código fuente TeX de la presentación Beamer
├── articulo_ieee.pdf            # Artículo científico en formato IEEEtran (PDF)
├── articulo_ieee.docx           # Artículo científico en formato IEEE (Word)
├── articulo_ieee.tex            # Código fuente TeX del artículo científico
├── notebooks/                   # Jupyter Notebooks ordenados secuencialmente por fases CRISP-DM
│   ├── 01_EDA.ipynb             # Exploración de datos y distribución de clases
│   ├── 02_preprocesamiento.ipynb# Limpieza, RobustScaler, selección por Gini y Borderline-SMOTE1
│   ├── 03_modelos_base.ipynb    # Entrenamiento OOF 3-Fold y extracción de Entropía de Shannon
│   ├── 04_modelo_hibrido.ipynb  # Ensamblado del Stacking Nivel 1 y búsqueda con GridSearchCV
│   └── 05_resultados_finales.ipynb# Evaluación final, perfilado computacional y test de Wilcoxon
├── src/                         # Módulos Python parametrizados para ejecución en consola
│   ├── 08_preprocesamiento_pipeline.py
│   ├── 09_modelos_base_pipeline.py
│   ├── 10_modelo_hibrido_pipeline.py
│   ├── 11_resultados_finales_pipeline.py
│   ├── 13_export_onnx_pipeline.py
│   └── sampling.py
├── results/                     # Metadatos JSON y gráficos de resultados empíricos
│   ├── figures/                 # Gráficos de matrices de confusión, boxplots e importancias
│   ├── label_mapping.json       # Mapeo numérico de clases
│   ├── selected_features.json   # Vector de 29 características seleccionadas
│   └── prueba_wilcoxon.json     # Resultados de la prueba no paramétrica
├── models/                      # Configuraciones de hiperparámetros y artefactos ONNX
├── requirements.txt             # Dependencias del entorno virtual
├── .gitignore                   # Exclusión de datasets masivos y archivos temporales
└── README.md                    # Documentación principal del repositorio
```

---

## 🚀 Resultados Empíricos

Evaluados sobre la partición de prueba aislada e independiente ($N_{\mathrm{test}} = 377,383$ muestras) del dataset **CICIoT2023**:

| Arquitectura de Clasificación | F1-Macro | Accuracy | Precision | Recall | Latencia ($\mu s$) | Memoria RAM (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Regresión Logística Base | 0.4220 | 0.6120 | 0.4105 | 0.4350 | - | - |
| Árbol de Decisión CART | 0.6651 | 0.8124 | 0.6720 | 0.6590 | - | - |
| Random Forest Base | 0.6791 | 0.8350 | 0.6840 | 0.6750 | 3.24 | 184.20 |
| LightGBM Base | 0.6953 | 0.8420 | 0.7010 | 0.6910 | 42.08 | 195.10 |
| XGBoost GPU Base | 0.6960 | 0.8435 | 0.7025 | 0.6920 | 1.34 | 142.50 |
| Random Forest + Borderline-SMOTE | 0.6850 | 0.8290 | 0.6890 | 0.6820 | - | - |
| XGBoost GPU + Borderline-SMOTE | 0.6995 | 0.8460 | 0.7050 | 0.6950 | - | - |
| **Stacking Híbrido (Propuesto)** | **0.7018** | **0.8485** | **0.7075** | **0.6980** | **8.65** | **257.89** |

---

## 🛠️ Instalación y Configuración

### 1. Clonar el Repositorio
```bash
git clone https://github.com/carmenNieves6478/DETECCI-N-DE-ATAQUES-MINORITARIOS-EN-REDES-IoT-MEDIANTE-UN-MODELO-H-BRIDO.git
cd DETECCI-N-DE-ATAQUES-MINORITARIOS-EN-REDES-IoT-MEDIANTE-UN-MODELO-H-BRIDO
```

### 2. Crear Entorno Virtual e Instalar Dependencias
```bash
conda create -n tesis_iot python=3.10.13 -y
conda activate tesis_iot
pip install -r requirements.txt
```

---

## 📄 Licencia

Este proyecto está bajo la Licencia MIT.
