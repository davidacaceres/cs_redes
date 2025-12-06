
import os
import shutil
from pathlib import Path
import json

def alinear_archivos(base_dir: Path):
    sitio_dir = base_dir / "sitio"
    banderas_dir = sitio_dir / "banderas"
    info_dir = sitio_dir / "info"
    
    data_dir = base_dir / "data"
    kujala_dir = data_dir / "kujala" / "procesado" # Assuming structure
    metro51_dir = data_dir / "51_metro_networks"
    
    # 1. Obtener nombres esperados
    nombres_esperados = set()
    
    # Kujala
    if kujala_dir.exists():
        for item in kujala_dir.iterdir():
            if item.is_dir():
                nombres_esperados.add(f"kujala_{item.name}")
                
    # Metro51
    if metro51_dir.exists():
        for item in metro51_dir.glob("*.json"):
            nombres_esperados.add(item.stem)
            
    print(f"Esperados: {len(nombres_esperados)} identificadores de grafos.")
    
    # 2. Mapear archivos existentes
    # Estructura actual: <ciudad>-L.png / .html
    
    def procesar_carpeta(origen: Path, extension_origen: str, extension_destino: str):
        if not origen.exists():
            return
            
        archivos = list(origen.glob(f"*{extension_origen}"))
        print(f"Procesando {len(archivos)} archivos en {origen.name}...")
        
        for archivo in archivos:
            nombre_base = archivo.stem # ej: amsterdam-L
            
            # Quitar -L si existe
            nombre_limpio = nombre_base.replace("-L", "").lower()
            
            # Buscar coincidencia en nombres_esperados
            coincidencias = []
            for esperado in nombres_esperados:
                # Caso 1: Coincidencia exacta (ej: amsterdam -> kujala_amsterdam)
                if esperado == f"kujala_{nombre_limpio}":
                    coincidencias.append(esperado)
                
                # Caso 2: Contiene el nombre (ej: buenosaires -> argentina-buenosaires)
                elif nombre_limpio in esperado.split("-"):
                    coincidencias.append(esperado)
            
            # Copiar archivo a nuevos nombres
            if coincidencias:
                for destino_nombre in coincidencias:
                    destino_path = origen / f"{destino_nombre}{extension_destino}"
                    if not destino_path.exists():
                        print(f"  Mapping {archivo.name} -> {destino_path.name}")
                        shutil.copy(archivo, destino_path)
                    else:
                        print(f"  Skipping {destino_path.name} (exists)")
            else:
                print(f"  No match for {nombre_base}")

    procesar_carpeta(banderas_dir, ".png", ".png")
    procesar_carpeta(info_dir, ".html", ".md") # Rename .html to .md

if __name__ == "__main__":
    base_dir = Path(".")
    alinear_archivos(base_dir)
