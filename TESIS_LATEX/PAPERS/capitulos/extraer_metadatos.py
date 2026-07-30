import os
import re
from pathlib import Path
from pypdf import PdfReader
import pdfplumber

# Rutas de las carpetas
PAPERS_DIR = r"e:\PROYECTO DE INVESTIGACION\BORRADOR DE TESIS\PAPERS"
CARPETAS = ["Q1", "Q2", "Q3", "Otros", "TESIS"]

def extraer_metadatos_pdf(ruta_pdf):
    """Extrae metadatos de un PDF"""
    metadatos = {
        "archivo": os.path.basename(ruta_pdf),
        "titulo": "",
        "autores": "",
        "resumen": "",
        "doi": "",
        "isbn": "",
        "ano": "",
        "editorial": "",
        "volumen": "",
        "url": ""
    }
    
    try:
        # Intentar con PyPDF primero para metadatos
        with open(ruta_pdf, 'rb') as f:
            pdf_reader = PdfReader(f)
            
            # Obtener metadatos del PDF
            if pdf_reader.metadata:
                metadatos["titulo"] = pdf_reader.metadata.get("/Title", "").strip() or ""
                metadatos["autores"] = pdf_reader.metadata.get("/Author", "").strip() or ""
            
            # Extraer texto de primera página para encontrar información
            if len(pdf_reader.pages) > 0:
                primera_pagina = pdf_reader.pages[0]
                texto_primera = primera_pagina.extract_text()
            else:
                texto_primera = ""
        
        # Usar pdfplumber para extraer más texto y buscar metadatos
        with pdfplumber.open(ruta_pdf) as pdf:
            # Extraer texto de primeras páginas
            texto_completo = ""
            for i, page in enumerate(pdf.pages[:3]):  # Primeras 3 páginas
                texto_completo += page.extract_text() + "\n"
            
            # Buscar resumen (Abstract)
            match_abstract = re.search(
                r"(?:Abstract|ABSTRACT|Resumen|RESUMEN)[:\s]+(.*?)(?:1\.|Introduction|INTRODUCTION|Keywords|KEYWORDS)",
                texto_completo,
                re.IGNORECASE | re.DOTALL
            )
            if match_abstract:
                resumen = match_abstract.group(1).strip()
                metadatos["resumen"] = " ".join(resumen.split())[:500]  # Primeros 500 caracteres
            
            # Buscar DOI
            match_doi = re.search(
                r"(?:doi|DOI)[:\s]+([^\s\n]+)",
                texto_completo,
                re.IGNORECASE
            )
            if match_doi:
                metadatos["doi"] = match_doi.group(1).strip()
            
            # Buscar ISBN
            match_isbn = re.search(
                r"(?:ISBN|isbn)[:\s]+([0-9\-X]+)",
                texto_completo,
                re.IGNORECASE
            )
            if match_isbn:
                metadatos["isbn"] = match_isbn.group(1).strip()
            
            # Buscar año
            match_ano = re.search(
                r"(?:20\d{2}|19\d{2})",
                texto_completo
            )
            if match_ano:
                metadatos["ano"] = match_ano.group(0)
            
            # Buscar URL
            match_url = re.search(
                r"https?://[^\s\n]+",
                texto_completo
            )
            if match_url:
                metadatos["url"] = match_url.group(0)
            
            # Si no hay título en metadatos, intentar extraer del texto
            if not metadatos["titulo"]:
                lines = texto_completo.split('\n')
                for line in lines[:10]:
                    if len(line.strip()) > 20 and len(line.strip()) < 200:
                        metadatos["titulo"] = line.strip()
                        break
            
            # Si no hay autores en metadatos, intentar extraer
            if not metadatos["autores"]:
                for line in texto_completo.split('\n')[:15]:
                    if any(word in line.lower() for word in ["author", "by", "autores"]):
                        metadatos["autores"] = line.strip()
                        break
    
    except Exception as e:
        metadatos["error"] = str(e)
    
    return metadatos

def main():
    """Procesa todos los PDFs y genera archivo de salida"""
    resultados = []
    
    for carpeta in CARPETAS:
        ruta_carpeta = os.path.join(PAPERS_DIR, carpeta)
        if not os.path.exists(ruta_carpeta):
            continue
        
        print(f"Procesando carpeta: {carpeta}")
        archivos_pdf = [f for f in os.listdir(ruta_carpeta) if f.endswith('.pdf')]
        
        for archivo in sorted(archivos_pdf):
            ruta_completa = os.path.join(ruta_carpeta, archivo)
            print(f"  Extrayendo: {archivo}")
            metadatos = extraer_metadatos_pdf(ruta_completa)
            resultados.append((carpeta, metadatos))
    
    # Generar archivo de salida
    ruta_salida = os.path.join(PAPERS_DIR, "METADATOS_ARTICULOS.txt")
    
    with open(ruta_salida, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("METADATOS DE ARTÍCULOS - PROYECTO DE INVESTIGACIÓN\n")
        f.write("=" * 100 + "\n\n")
        
        contador = 1
        for carpeta, meta in resultados:
            f.write(f"[{contador}] CARPETA: {carpeta} | ARCHIVO: {meta['archivo']}\n")
            f.write("-" * 100 + "\n")
            
            f.write(f"TÍTULO:      {meta['titulo']}\n")
            f.write(f"AUTORES:     {meta['autores']}\n")
            f.write(f"AÑO:         {meta['ano']}\n")
            f.write(f"DOI:         {meta['doi']}\n")
            f.write(f"ISBN:        {meta['isbn']}\n")
            f.write(f"URL:         {meta['url']}\n")
            f.write(f"RESUMEN:     {meta['resumen']}\n")
            
            if "error" in meta:
                f.write(f"ERROR:       {meta['error']}\n")
            
            f.write("\n" + "=" * 100 + "\n\n")
            contador += 1
    
    print(f"\n✓ Archivo generado: {ruta_salida}")
    print(f"✓ Total de artículos procesados: {len(resultados)}")

if __name__ == "__main__":
    main()
