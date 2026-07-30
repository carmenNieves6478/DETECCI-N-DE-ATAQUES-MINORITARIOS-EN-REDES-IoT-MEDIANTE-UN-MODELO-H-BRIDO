import json
import os

def fix_notebook():
    notebook_path = "notebooks/01_EDA.ipynb"
    if not os.path.exists(notebook_path):
        print(f"Error: {notebook_path} no encontrado.")
        return
        
    print(f"Cargando {notebook_path}...")
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
        
    updated = False
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            if 'df = pd.read_csv(df_path)' in source and 'os.path.exists' not in source:
                print("Celda de carga de datos encontrada. Insertando lógica de consolidación...")
                cell["source"] = [
                    "# Cargar o consolidar el dataset si no existe\n",
                    "import os\n",
                    "import pandas as pd\n",
                    "import numpy as np\n",
                    "from tqdm.notebook import tqdm\n",
                    "\n",
                    "df_path = \"data/processed/dataset_consolidado.csv\"\n",
                    "\n",
                    "if not os.path.exists(df_path):\n",
                    "    print(\"El dataset consolidado no existe en data/processed/. Iniciando consolidación de 63 archivos raw...\")\n",
                    "    raw_dir = \"data/raw\"\n",
                    "    processed_dir = \"data/processed\"\n",
                    "    os.makedirs(processed_dir, exist_ok=True)\n",
                    "    \n",
                    "    files = sorted([f for f in os.listdir(raw_dir) if f.endswith(\".csv\")])\n",
                    "    samples_per_class_per_file = 2000\n",
                    "    seed = 42\n",
                    "    np.random.seed(seed)\n",
                    "    \n",
                    "    chunks = []\n",
                    "    for f in tqdm(files, desc=\"Procesando archivos\"):\n",
                    "        file_path = os.path.join(raw_dir, f)\n",
                    "        df_temp = pd.read_csv(file_path)\n",
                    "        sampled_dfs = []\n",
                    "        for label, group in df_temp.groupby('Label'):\n",
                    "            if len(group) > samples_per_class_per_file:\n",
                    "                sampled_group = group.sample(n=samples_per_class_per_file, random_state=seed)\n",
                    "                sampled_dfs.append(sampled_group)\n",
                    "            else:\n",
                    "                sampled_dfs.append(group)\n",
                    "        if sampled_dfs:\n",
                    "            chunks.append(pd.concat(sampled_dfs, axis=0))\n",
                    "            \n",
                    "    print(\"Concatenando todos los fragmentos...\")\n",
                    "    df_final = pd.concat(chunks, axis=0, ignore_index=True)\n",
                    "    print(f\"Guardando dataset consolidado en: {df_path}...\")\n",
                    "    df_final.to_csv(df_path, index=False)\n",
                    "    print(\"¡Consolidación completada y guardada con éxito!\")\n",
                    "\n",
                    "print(\"Cargando dataset consolidado...\")\n",
                    "df = pd.read_csv(df_path)\n",
                    "print(f\"¡Cargado con éxito! Dimensiones: {df.shape[0]:,} filas, {df.shape[1]} columnas.\")"
                ]
                updated = True
                break
                
    if updated:
        with open(notebook_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print("Notebook 01_EDA.ipynb actualizado correctamente.")
    else:
        print("No se requirieron cambios en el notebook o ya estaba actualizado.")

if __name__ == "__main__":
    fix_notebook()
