"""
preparar_datos
--------------

Módulo para descarga y preparación de datasets de redes de transporte público.

Este módulo proporciona funciones para:
- Descargar archivos desde URLs HTTP
- Extraer archivos comprimidos (ZIP)
- Organizar datasets en estructura de directorios

Ejemplos de uso
---------------

Descargar y extraer el dataset Kujala::

    from pathlib import Path
    from preparar_datos import descargar_archivo, extraer_zip
    
    # Descargar dataset
    url = "https://example.com/kujala_dataset.zip"
    zip_path = descargar_archivo(url, Path("data/kujala.zip"))
    
    # Extraer
    extraer_zip(zip_path, Path("data/kujala"))

Descargar múltiples ciudades::

    from preparar_datos import descargar_ciudades_kujala
    
    descargar_ciudades_kujala(
        directorio_destino=Path("data/kujala_zips"),
        ciudades=("adelaide", "berlin", "paris")
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple
import shutil

import requests


def descargar_archivo(url: str, destino: Path, tamano_chunk: int = 2 ** 20) -> Path:
    """Descarga un archivo vía HTTP y lo guarda en disco.
    
    Si el archivo ya existe en el destino, no se descarga nuevamente.
    
    Parámetros
    ----------
    url : str
        URL del archivo a descargar.
    destino : Path
        Ruta donde se guardará el archivo descargado.
    tamano_chunk : int, opcional
        Tamaño del chunk para descarga en streaming (por defecto 1 MB).
    
    Retorna
    -------
    Path
        Ruta donde se guardó el archivo.
    
    Excepciones
    -----------
    requests.HTTPError
        Si la descarga falla (código de estado HTTP no exitoso).
    
    Ejemplos
    --------
    >>> from pathlib import Path
    >>> url = "https://example.com/data.csv"
    >>> archivo = descargar_archivo(url, Path("data/datos.csv"))
    >>> print(f"Archivo descargado en: {archivo}")
    """
    destino.parent.mkdir(parents=True, exist_ok=True)
    if destino.exists():
        print(f"[INFO] Archivo ya existe, omitiendo descarga: {destino}")
        return destino
    
    print(f"[INFO] Iniciando descarga desde: {url}")
    print(f"[INFO] Destino: {destino}")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_bytes = 0
    with open(destino, "wb") as fh:
        for chunk in response.iter_content(chunk_size=tamano_chunk):
            if chunk:
                fh.write(chunk)
                total_bytes += len(chunk)
                # Mostrar progreso cada 10 MB
                if total_bytes % (10 * 1024 * 1024) < tamano_chunk:
                    print(f"[INFO] Descargados: {total_bytes / (1024 * 1024):.1f} MB")
    
    print(f"[INFO] Descarga completada: {total_bytes / (1024 * 1024):.1f} MB guardados en {destino}")
    return destino


def extraer_zip(ruta_zip: Path, directorio_destino: Path) -> None:
    """Extrae un archivo ZIP en un directorio.
    
    Si el directorio destino ya contiene archivos, no se extrae nuevamente
    para evitar sobrescribir datos existentes.
    
    Parámetros
    ----------
    ruta_zip : Path
        Ruta al archivo ZIP a extraer.
    directorio_destino : Path
        Directorio donde se extraerán los archivos.
    
    Ejemplos
    --------
    >>> from pathlib import Path
    >>> extraer_zip(Path("data/dataset.zip"), Path("data/dataset"))
    """
    import zipfile
    directorio_destino.mkdir(parents=True, exist_ok=True)
    
    if any(directorio_destino.iterdir()):
        print(f"[INFO] Directorio destino ya contiene archivos, omitiendo extracción: {directorio_destino}")
        return
    
    print(f"[INFO] Extrayendo archivo ZIP: {ruta_zip}")
    print(f"[INFO] Destino: {directorio_destino}")
    
    with zipfile.ZipFile(ruta_zip, "r") as zf:
        num_archivos = len(zf.namelist())
        print(f"[INFO] Extrayendo {num_archivos} archivos...")
        zf.extractall(directorio_destino)
    
    print(f"[INFO] Extracción completada: {num_archivos} archivos extraídos")


def descargar_ciudades_kujala(
    directorio_destino: Path,
    id_registro: str = "1186215",
    ciudades: Tuple[str, ...] = ()
) -> None:
    """Descarga los archivos ZIP del dataset Kujala para las ciudades especificadas.

    Parámetros
    ----------
    directorio_destino : Path
        Directorio donde se guardarán los archivos ZIP.
    id_registro : str, opcional
        ID del registro de Zenodo que contiene los archivos (por defecto "1186215").
    ciudades : tuple de str, opcional
        Nombres de las ciudades a descargar. Si está vacío, descarga todas las ciudades.
    
    Ejemplos
    --------
    >>> from pathlib import Path
    >>> descargar_ciudades_kujala(
    ...     Path("data/kujala_zips"),
    ...     ciudades=("adelaide", "berlin")
    ... )
    """
    directorio_destino.mkdir(parents=True, exist_ok=True)
    if not ciudades:
        # Lista completa del artículo original (25 ciudades)
        ciudades = (
            "adelaide", "belfast", "berlin", "bordeaux", "brisbane",
            "canberra", "detroit", "dublin", "grenoble", "helsinki",
            "kuopio", "lisbon", "luxembourg", "melbourne", "nantes",
            "palermo", "paris", "prague", "rennes", "rome",
            "sydney", "toulouse", "turku", "venice", "winnipeg",
        )
    
    print(f"[INFO] Descargando {len(ciudades)} ciudades del dataset Kujala")
    for i, ciudad in enumerate(ciudades, 1):
        url = f"https://zenodo.org/records/{id_registro}/files/{ciudad}.zip?download=1"
        ruta_zip = directorio_destino / f"{ciudad}.zip"
        print(f"\n[INFO] Descargando ciudad {i}/{len(ciudades)}: {ciudad}")
        try:
            descargar_archivo(url, ruta_zip)
        except Exception as exc:
            print(f"[ERROR] No se pudo descargar {ciudad}: {exc}")
    
    print(f"\n[INFO] Descarga de ciudades Kujala completada")


def extraer_datasets(
    dir_zip_kujala: Path,
    dir_extraer_kujala: Path,
    ruta_zip_metro: Path,
    dir_extraer_metro: Path
) -> None:
    """Extrae todos los archivos ZIP descargados para ambos datasets.

    Esta función extrae las ciudades de Kujala y reorganiza los archivos en una
    estructura limpia, copiando solo los archivos CSV necesarios (network_nodes.csv
    y network_combined.csv) a un directorio procesado.

    Parámetros
    ----------
    dir_zip_kujala : Path
        Directorio que contiene los ZIPs de las ciudades Kujala.
    dir_extraer_kujala : Path
        Directorio donde se crearán los directorios de ciudades.
    ruta_zip_metro : Path
        Ruta al archivo ZIP del dataset de 51 metros.
    dir_extraer_metro : Path
        Directorio donde se extraerá el dataset de 51 metros.
    
    Ejemplos
    --------
    >>> from pathlib import Path
    >>> extraer_datasets(
    ...     Path("data/kujala_zips"),
    ...     Path("data/kujala"),
    ...     Path("data/metro51.zip"),
    ...     Path("data/metro51")
    ... )
    """
    print("[INFO] Iniciando extracción de datasets")
    
    # Extraer ciudades Kujala y reorganizar
    preprocesamiento_dir = dir_extraer_kujala.parent / "kujala" / "preprocesamiento"
    procesado_dir = dir_extraer_kujala.parent / "kujala" / "procesado"
    
    archivos_zip = list(dir_zip_kujala.glob("*.zip"))
    print(f"[INFO] Encontrados {len(archivos_zip)} archivos ZIP de ciudades Kujala")
    
    for i, zip_file in enumerate(archivos_zip, 1):
        ciudad_nombre = zip_file.stem
        print(f"\n[INFO] Extrayendo ciudad {i}/{len(archivos_zip)}: {ciudad_nombre}")
        
        preproc_ciudad_dir = preprocesamiento_dir / ciudad_nombre
        
        # Extraer a ubicación de preprocesamiento
        extraer_zip(zip_file, preproc_ciudad_dir)
        
        # Crear estructura de directorio limpia en procesado
        ciudad_limpia_dir = procesado_dir / ciudad_nombre
        ciudad_limpia_dir.mkdir(parents=True, exist_ok=True)
        
        # Encontrar y copiar solo los archivos CSV que necesitamos
        # Los archivos típicamente están en preproc_ciudad_dir/ciudad_nombre/
        directorio_fuente = preproc_ciudad_dir / ciudad_nombre
        if directorio_fuente.exists():
            archivo_nodos = directorio_fuente / "network_nodes.csv"
            archivo_aristas = directorio_fuente / "network_combined.csv"
            
            if archivo_nodos.exists():
                shutil.copy2(archivo_nodos, ciudad_limpia_dir / "network_nodes.csv")
                print(f"[INFO] Copiado: network_nodes.csv")
            if archivo_aristas.exists():
                shutil.copy2(archivo_aristas, ciudad_limpia_dir / "network_combined.csv")
                print(f"[INFO] Copiado: network_combined.csv")
    
    # Extraer 51 metros
    if ruta_zip_metro and ruta_zip_metro.exists():
        print(f"\n[INFO] Extrayendo dataset de 51 metros")
        extraer_zip(ruta_zip_metro, dir_extraer_metro)
    
    print("\n[INFO] Extracción de datasets completada")
