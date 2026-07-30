import json
import os

def fix_tqdm():
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
            if 'from tqdm.notebook import tqdm' in source:
                print("Reemplazando tqdm.notebook por tqdm estándar para evitar error de ipywidgets...")
                # Reemplazar la línea exacta
                new_source = []
                for line in cell["source"]:
                    if 'from tqdm.notebook import tqdm' in line:
                        new_source.append("from tqdm import tqdm\n")
                    else:
                        new_source.append(line)
                cell["source"] = new_source
                updated = True
                break
                
    if updated:
        with open(notebook_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print("Notebook 01_EDA.ipynb actualizado con tqdm estándar.")
    else:
        print("No se requirieron cambios.")

if __name__ == "__main__":
    fix_tqdm()
