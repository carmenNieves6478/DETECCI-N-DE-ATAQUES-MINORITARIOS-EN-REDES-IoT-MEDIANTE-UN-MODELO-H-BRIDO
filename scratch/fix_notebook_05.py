import json
import os

def fix_notebook_05():
    notebook_path = "notebooks/05_resultados_finales.ipynb"
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
            # Reemplazar meta_decision_tree.pkl por meta_lightgbm.pkl
            if "meta_decision_tree.pkl" in source:
                print("Encontrada referencia a meta_decision_tree.pkl. Actualizando a meta_lightgbm.pkl...")
                new_source = []
                for line in cell["source"]:
                    new_line = line.replace("meta_decision_tree.pkl", "meta_lightgbm.pkl")
                    new_source.append(new_line)
                cell["source"] = new_source
                updated = True
                
    if updated:
        with open(notebook_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print("Notebook 05_resultados_finales.ipynb actualizado con éxito.")
    else:
        print("No se encontraron referencias a meta_decision_tree.pkl.")

if __name__ == "__main__":
    fix_notebook_05()
