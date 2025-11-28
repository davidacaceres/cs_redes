#!/usr/bin/env python3
"""
assignment3_script
===================

Este script reproduce los experimentos de la Tarea 3 del curso de
Ciencia de Redes sin necesidad de un cuaderno Jupyter. Su propósito
es descargar y procesar conjuntos de datos de redes de transporte
público, calcular métricas de robustez y generar resúmenes en
formato tabular. El código utiliza únicamente dependencias
incluidas en ``assignment3_module.py``, lo que evita requerir
bibliotecas de grafos externas como ``networkx``.

Contenido
---------

1. **Resumen del artículo** – Describe brevemente el enfoque y
   conclusiones principales de Wang et al. (2017) “Multi‐criteria
   robustness analysis of metro networks”.
2. **Descarga de datos** – Funciones para obtener y descomprimir los
   conjuntos de datos de 25 ciudades (Kujala et al.) y 51 redes de
   metro. Estos archivos se alojan respectivamente en Zenodo y Google
   Drive. Si ya existen en disco, no se vuelven a descargar.
3. **Cálculo de métricas** – Carga las redes con ``assignment3_module``
   y calcula métricas básicas (número de nodos, enlaces, grado
   promedio), indicadores teóricos de robustez ``r_T`` y
   ``C_G``, y una métrica numérica simple de robustez basada en
   la eliminación de nodos de mayor grado o al azar.
4. **Generación de resumen** – Produce un `DataFrame` con los
   resultados para cada red y lo imprime en pantalla. Puede guardarse
   en CSV si se desea.

Resumen del artículo
--------------------

El estudio de Wang et al. (2017) analiza la robustez de 33 redes
de metro mediante una combinación de métricas teóricas y
numéricas【282482872483239†L121-L162】【282482872483239†L163-L209】. Dos de las métricas teóricas
principales son:

Indicador de robustez r_T**: se define como \\((L - N + 1)/N\\),
  donde \\(L\\) es el número de enlaces y \\(N\\) el número de nodos. Este
  indicador mide la densidad de ciclos: valores altos indican más
  rutas alternativas, pero se atenúa conforme la red crece【282482872483239†L121-L162】.
Conductancia efectiva C_G**: se deriva de la resistencia efectiva
  y combina redundancia y longitud de caminos. Se calcula como
  \\((N-1)/R_G\\), con \\(R_G\\) la suma de inversos de los valores
  propios no nulos del laplaciano de la componente gigante. Redes
  con caminos cortos y muchas alternativas presentan valores altos
  de C_G【282482872483239†L163-L209】.

Los autores complementan estas métricas con experimentos de
simulación donde eliminan nodos al azar y en orden decreciente de
grado, obteniendo umbrales críticos de colapso: \\(f_{90\\%}\\) (fracción
de nodos cuya eliminación reduce la componente gigante al 90 % de
su tamaño original) y \\(f_c\\) (fracción necesaria para desconectar la
red)【282482872483239†L238-L335】. Encuentran que métricas basadas en ciclos (r_T) y en
eficiencia (C_G) pueden ser antagónicas: redes grandes con muchas
rutas alternativas (p.ej. Tokio) presentan alto r_T pero caminos
largos; otras redes pequeñas con estructuras ramificadas (p.ej.
Roma) alcanzan alta C_G debido a la corta longitud de sus trayectos
【282482872483239†L389-L498】【282482872483239†L838-L871】. La correlación entre C_G y los
umbrales f_{90\\%} es fuerte y positiva, mientras que r_T se correlaciona
mejor con f_c【282482872483239†L838-L871】. Los autores concluyen que una evaluación de
robustez completa debe considerar múltiples criterios【282482872483239†L874-L901】.

Referencias (APA 7)
-------------------

Wang, J., Sun, X., Li, X., Huang, Y., & Ding, J. (2017).
Multi‐criteria robustness analysis of metro networks.
*Physica A*, 474, 19–31. https://doi.org/10.1016/j.physa.2017.01.013

Kujala, R., Weckström, C., Malkoç, G., & Saramäki, J. (2018).
A collection of public transport network data sets for 25 cities.
*Scientific Data*, 5, 180089. https://doi.org/10.1038/sdata.2018.89


"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple


# Import helper functions from the module supplied with this assignment.
from tarea3_module import (
    download_file,
    download_from_drive,
    extract_zip,
    load_kujala_dataset,
    load_metro51_from_folder,
    compute_dataset_summary,
    SimpleGraph,
)


def download_kujala_cities(dest_dir: Path, record_id: str = "1186215", cities: Tuple[str, ...] = ()) -> None:
    """Download the Kujala dataset ZIP files for the given cities.

    Parameters
    ----------
    dest_dir : Path
        Directory where the ZIP files will be stored.
    record_id : str, optional
        Zenodo record ID containing the city archives (default ``"1186215"``).
    cities : tuple of str, optional
        Names of cities to download. If empty, all cities in the record are
        downloaded.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    if not cities:
        # Complete list from the original article (25 cities)
        cities = (
            "adelaide", "belfast", "berlin", "bordeaux", "brisbane",
            "canberra", "detroit", "dublin", "grenoble", "helsinki",
            "kuopio", "lisbon", "luxembourg", "melbourne", "nantes",
            "palermo", "paris", "prague", "rennes", "rome",
            "sydney", "toulouse", "turku", "venice", "winnipeg",
        )
    for city in cities:
        url = f"https://zenodo.org/records/{record_id}/files/{city}.zip?download=1"
        zip_path = dest_dir / f"{city}.zip"
        print(f"Descargando {city}.zip...")
        try:
            download_file(url, zip_path)
        except Exception as exc:
            print(f"No se pudo descargar {city}: {exc}")


def download_metro51(dest_zip: Path, file_id: str) -> None:
    """Download the 51 metro dataset from Google Drive using its file ID.

    Parameters
    ----------
    dest_zip : Path
        Destination ZIP file path.
    file_id : str
        Google Drive file ID (extract it from the share link).
    """
    dest_zip.parent.mkdir(parents=True, exist_ok=True)
    if dest_zip.exists():
        print(f"Archivo {dest_zip.name} ya descargado.")
        return
    print("Descargando dataset de 51 metros...")
    try:
        download_from_drive(file_id, dest_zip)
    except Exception as exc:
        print(f"Error al descargar el dataset de 51 metros: {exc}")


def extract_datasets(kujala_zip_dir: Path, kujala_extract_dir: Path, metro_zip_path: Path, metro_extract_dir: Path) -> None:
    """Extract all downloaded ZIP archives for both datasets.

    Parameters
    ----------
    kujala_zip_dir : Path
        Directory containing the ZIPs for Kujala cities.
    kujala_extract_dir : Path
        Directory where the city directories will be created.
    metro_zip_path : Path
        Path to the downloaded 51 metros ZIP file.
    metro_extract_dir : Path
        Directory where the 51 metros dataset will be extracted.
    """
    import shutil
    
    # Extract Kujala cities and reorganize
    preprocesamiento_dir = kujala_extract_dir.parent / "kujala" / "preprocesamiento"
    procesado_dir = kujala_extract_dir.parent / "kujala" / "procesado"
    
    for zip_file in kujala_zip_dir.glob("*.zip"):
        city_name = zip_file.stem
        preproc_city_dir = preprocesamiento_dir / city_name
        
        # Extract to preprocesamiento location
        extract_zip(zip_file, preproc_city_dir)
        
        # Create clean directory structure in procesado
        clean_city_dir = procesado_dir / city_name
        clean_city_dir.mkdir(parents=True, exist_ok=True)
        
        # Find and copy only the CSV files we need
        # Files are typically in preproc_city_dir/city_name/
        source_dir = preproc_city_dir / city_name
        if source_dir.exists():
            nodes_file = source_dir / "network_nodes.csv"
            edges_file = source_dir / "network_combined.csv"
            
            if nodes_file.exists():
                shutil.copy2(nodes_file, clean_city_dir / "network_nodes.csv")
            if edges_file.exists():
                shutil.copy2(edges_file, clean_city_dir / "network_combined.csv")
    
    # Extract 51 metros
    if metro_zip_path and metro_zip_path.exists():
        extract_zip(metro_zip_path, metro_extract_dir)


def load_datasets(
    kujala_dir: Path,
    metro51_dir: Path,
    use_kujala: bool = True,
    use_metro51: bool = False,
) -> Dict[str, Dict[str, SimpleGraph]]:
    """Load available networks from the specified directories.

    Returns a dictionary with keys ``"kujala"`` and ``"metro51"``, each
    mapping to another dictionary of network name -> graph. If a
    dataset is disabled by the flags, the corresponding entry may be
    empty.
    """
    print("load data set ",kujala_dir)
    result: Dict[str, Dict[str, SimpleGraph]] = {}
    if use_kujala and kujala_dir.exists():
        graphs, _ = load_kujala_dataset(kujala_dir)
        result["kujala"] = graphs
    else:
        result["kujala"] = {}
    if use_metro51 and metro51_dir.exists():
        graphs, _ = load_metro51_from_folder(metro51_dir)
        result["metro51"] = graphs
    else:
        result["metro51"] = {}
    return result


def main(argv: list[str] | None = None) -> None:
    """Punto de entrada principal.

    Este comando permite descargar los datos, extraerlos, cargar
    redes y calcular métricas. El uso típico consiste en ejecutar
    ``python assignment3_script.py --download --compute`` para
    descargar los datasets (si no están presentes) y generar un
    resumen de métricas.
    """
    parser = argparse.ArgumentParser(description="Ejecuta la Tarea 3 en modo script.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Carpeta base para guardar los datos descargados.")
    parser.add_argument("--download", action="store_true", default=True, help="Descargar datasets desde Internet.")
    parser.add_argument("--compute", action="store_true", default=True, help="Calcular métricas y generar resumen.")
    parser.add_argument("--include-metro51", action="store_true", default=False , help="Incluir el dataset de 51 metros si está disponible.")
    parser.add_argument("--metro51-id", type=str, default="https://drive.google.com/file/d/1d3HiojoRLHd5o56dGwpgkrSIIXgFbJ0q/view?usp=drive_link", help="ID de Google Drive para el dataset de 51 metros.")
    parser.add_argument("--output-csv", type=Path, default=None, help="Ruta para escribir el resumen en CSV.")
    args = parser.parse_args(argv)

    base_dir = args.data_dir
    kujala_zip_dir = base_dir / "kujala_zips"
    kujala_extract_dir = base_dir / "kujala"
    metro_zip_path = base_dir / "metro51.zip"
    metro_extract_dir = base_dir / "metro51"

    if args.download:
        # Descargar 25 ciudades
        download_kujala_cities(kujala_zip_dir)
        # Descargar 51 metros solo si se solicita y se proporciona un ID
    if args.include_metro51 and args.metro51_id:
        download_metro51(metro_zip_path, args.metro51_id)
        # Extraer todo
    extract_datasets(kujala_zip_dir, kujala_extract_dir, metro_zip_path, metro_extract_dir)

    if args.compute:
        # Cargar redes disponibles desde el directorio procesado
        datasets = load_datasets(
            kujala_dir=base_dir / "kujala" / "procesado",
            metro51_dir=metro_extract_dir,
            use_kujala=True,
            use_metro51=args.include_metro51,
        )
        # Unir todos los grafos
        all_graphs: Dict[str, SimpleGraph] = {}
        for group, gdict in datasets.items():
            prefix = "K_" if group == "kujala" else "M_"
            for name, G in gdict.items():
                all_graphs[f"{prefix}{name}"] = G
        if not all_graphs:
            print("No se encontraron redes. Asegúrate de descargar los datos o revisar las rutas.")
            return
        # Calcular métricas y robustez
        summary = compute_dataset_summary(
            all_graphs,
            frac_remove=0.2,
            random_runs=10,
            seed=42,
        )
        print("\nResumen de métricas:")
        print(summary.to_string(index=False))
        if args.output_csv is not None:
            summary.to_csv(args.output_csv, index=False)
            print(f"Resumen guardado en {args.output_csv}")


if __name__ == "__main__":
    main()