"""
dataset_loader
---------------

Módulo para descarga y preparación de datasets de redes de transporte público.

Este módulo proporciona funciones para:
- Descargar archivos desde URLs HTTP y Google Drive
- Extraer archivos comprimidos (ZIP)
- Construir grafos desde datasets de ciudades (formato Kujala)
- Cargar datasets de redes de metro (51 metros)

Todos los grafos se construyen usando la clase SimpleGraph del módulo principal,
que no depende de bibliotecas externas de grafos como networkx.

Ejemplos de uso
---------------

Descargar y cargar el dataset Kujala::

    from pathlib import Path
    from dataset_loader import descargar_archivo, extraer_zip, cargar_dataset_kujala
    
    # Descargar dataset
    url = "https://example.com/kujala_dataset.zip"
    zip_path = descargar_archivo(url, Path("data/kujala.zip"))
    
    # Extraer
    extraer_zip(zip_path, Path("data/kujala"))
    
    # Cargar grafos
    grafos, nodos = cargar_dataset_kujala(Path("data/kujala"))
    
    # Usar los grafos
    for ciudad, grafo in grafos.items():
        print(f"{ciudad}: {grafo.number_of_nodes()} nodos")

Cargar dataset de metros::

    from dataset_loader import cargar_metros_desde_carpeta
    
    grafos, nodos = cargar_metros_desde_carpeta(Path("data/metro51"))
    for nombre, grafo in grafos.items():
        print(f"{nombre}: {grafo.number_of_edges()} aristas")
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Any

import pandas as pd

try:
    import gdown  # type: ignore
    _HAS_GDOWN = True
except Exception:
    _HAS_GDOWN = False

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


def construir_grafo_desde_ciudad_kujala(
    directorio_ciudad: Path
) -> Tuple[Any, pd.DataFrame]:
    """Construye un SimpleGraph desde un directorio de ciudad del dataset Kujala.
    
    Cada directorio de ciudad debe contener los archivos:
    - network_nodes.csv: nodos de la red (paradas) con separador ';'
    - network_combined.csv: aristas de la red con separador ';'
    
    La función lee el archivo de nodos, mapea columnas según nombres posibles
    para identificador, latitud, longitud y nombre, y construye un grafo no
    dirigido de paradas. Los atributos de nodos se preservan en un DataFrame.
    
    Parámetros
    ----------
    directorio_ciudad : Path
        Directorio que contiene los archivos CSV de la ciudad.
    
    Retorna
    -------
    grafo : SimpleGraph
        Grafo construido con los nodos y aristas de la ciudad.
    nodos_df : pd.DataFrame
        DataFrame con los atributos de cada nodo.
    
    Excepciones
    -----------
    FileNotFoundError
        Si faltan los archivos requeridos en el directorio.
    ValueError
        Si no se encuentran las columnas necesarias en los CSV.
    
    Ejemplos
    --------
    >>> from pathlib import Path
    >>> grafo, nodos = construir_grafo_desde_ciudad_kujala(Path("data/kujala/adelaide"))
    >>> print(f"Nodos: {grafo.number_of_nodes()}, Aristas: {grafo.number_of_edges()}")
    """
    # Importar SimpleGraph del módulo principal
    from tarea3_module import SimpleGraph
    
    print(f"[INFO] Construyendo grafo para ciudad: {directorio_ciudad.name}")
    
    nodos_csv = directorio_ciudad / "network_nodes.csv"
    aristas_csv = directorio_ciudad / "network_combined.csv"

    print(f"[INFO] Buscando archivo de nodos: {nodos_csv}")
    print(f"[INFO] Buscando archivo de aristas: {aristas_csv}")
    
    if not nodos_csv.exists() or not aristas_csv.exists():
        print(f"[ERROR] Archivo no encontrado en {directorio_ciudad}")
        raise FileNotFoundError(f"Faltan archivos requeridos en {directorio_ciudad}")
    
    print(f"[INFO] Leyendo archivo de nodos...")
    nodos_df = pd.read_csv(nodos_csv, sep=";")
    print(f"[INFO] Leídos {len(nodos_df)} nodos")
    
    # Determinar columna de identificador
    id_col = None
    for candidato in ["stop_I", "node_I", "node_id", "stop_id", "stop_i"]:
        if candidato in nodos_df.columns:
            id_col = candidato
            break
    if id_col is None:
        print(f"[ERROR] No se encontró columna de identificador. Columnas disponibles: {list(nodos_df.columns)}")
        raise ValueError(f"No se encontró columna de identificador en {nodos_csv}")
    
    print(f"[INFO] Usando columna '{id_col}' como identificador de nodo")
    
    # Determinar columnas de atributos
    lat_col = next(
        (c for c in nodos_df.columns if c.lower() in {"lat", "latitude"}), None
    )
    lon_col = next(
        (c for c in nodos_df.columns if c.lower() in {"lon", "longitude"}), None
    )
    name_col = next(
        (c for c in nodos_df.columns if "name" in c.lower()), None
    )
    
    # Construir grafo
    print(f"[INFO] Construyendo grafo con nombre: kujala_{directorio_ciudad.name}")
    G = SimpleGraph(name=f"kujala_{directorio_ciudad.name}")
    atributos_nodos: List[Dict[str, Any]] = []
    
    print(f"[INFO] Agregando nodos al grafo...")
    for _, fila in nodos_df.iterrows():
        nid = fila[id_col]
        attrs: Dict[str, Any] = {}
        if lat_col is not None and pd.notna(fila[lat_col]):
            attrs["lat"] = float(fila[lat_col])
        if lon_col is not None and pd.notna(fila[lon_col]):
            attrs["lon"] = float(fila[lon_col])
        if name_col is not None and pd.notna(fila[name_col]):
            attrs["name"] = str(fila[name_col])
        G.add_node(nid, **attrs)
        atributos_nodos.append({"node_id": nid, **attrs})
    
    print(f"[INFO] {G.number_of_nodes()} nodos agregados al grafo")
    
    # Leer aristas
    print(f"[INFO] Leyendo archivo de aristas...")
    aristas_df = pd.read_csv(aristas_csv, sep=";")
    print(f"[INFO] Leídas {len(aristas_df)} aristas")
    
    from_col = None
    to_col = None
    
    for candidato in ["from_stop_I", "from_node_I", "from_id", "from_stop", "u"]:
        if candidato in aristas_df.columns:
            from_col = candidato
            break
    for candidato in ["to_stop_I", "to_node_I", "to_id", "to_stop", "v"]:
        if candidato in aristas_df.columns:
            to_col = candidato
            break
    
    if from_col is None or to_col is None:
        print(f"[ERROR] No se encontraron columnas de aristas. Columnas disponibles: {list(aristas_df.columns)}")
        raise ValueError(f"No se encontraron columnas de aristas en {aristas_csv}")
    
    print(f"[INFO] Usando columnas '{from_col}' -> '{to_col}' para aristas")
    
    print(f"[INFO] Agregando aristas al grafo...")
    aristas_agregadas = 0
    for _, fila in aristas_df.iterrows():
        u = fila[from_col]
        v = fila[to_col]
        if pd.isna(u) or pd.isna(v):
            continue
        # Convertir a int si es posible
        try:
            u_int = int(u)
            v_int = int(v)
        except Exception:
            u_int = u
            v_int = v
        G.add_edge(u_int, v_int)
        aristas_agregadas += 1
    
    print(f"[INFO] {aristas_agregadas} aristas agregadas al grafo")
    
    nodos_df_salida = pd.DataFrame(atributos_nodos)
    print(f"[INFO] Grafo construido exitosamente para {directorio_ciudad.name}: {G.number_of_nodes()} nodos, {G.number_of_edges()} aristas")
    return G, nodos_df_salida


def cargar_dataset_kujala(
    raiz: Path
) -> Tuple[Dict[str, Any], Dict[str, pd.DataFrame]]:
    """Carga todas las redes de ciudades desde el directorio del dataset Kujala.
    
    Itera sobre todos los subdirectorios en la ruta raíz e intenta construir
    un grafo para cada ciudad encontrada. Las ciudades que no puedan ser
    procesadas se omiten silenciosamente.
    
    Parámetros
    ----------
    raiz : Path
        Directorio raíz que contiene subdirectorios de ciudades.
    
    Retorna
    -------
    grafos : Dict[str, SimpleGraph]
        Diccionario con nombre de ciudad como clave y grafo como valor.
    nodos : Dict[str, pd.DataFrame]
        Diccionario con nombre de ciudad como clave y DataFrame de nodos como valor.
    
    Ejemplos
    --------
    >>> from pathlib import Path
    >>> grafos, nodos = cargar_dataset_kujala(Path("data/kujala"))
    >>> for ciudad, grafo in grafos.items():
    ...     print(f"{ciudad}: {grafo.number_of_nodes()} nodos")
    """
    print(f"[INFO] Iniciando carga del dataset Kujala desde: {raiz}")
    grafos: Dict[str, Any] = {}
    nodos: Dict[str, pd.DataFrame] = {}
    
    if not raiz.exists():
        print(f"[ADVERTENCIA] Directorio raíz no existe: {raiz}")
        return grafos, nodos
    
    directorios_ciudad = sorted([p for p in raiz.iterdir() if p.is_dir()])
    print(f"[INFO] Encontrados {len(directorios_ciudad)} directorios de ciudades")
    
    for i, dir_ciudad in enumerate(directorios_ciudad, 1):
        print(f"\n[INFO] Procesando ciudad {i}/{len(directorios_ciudad)}: {dir_ciudad.name}")
        try:
            G, df_nodos = construir_grafo_desde_ciudad_kujala(dir_ciudad)
            grafos[dir_ciudad.name] = G
            nodos[dir_ciudad.name] = df_nodos
            print(f"[INFO] Ciudad {dir_ciudad.name} cargada exitosamente")
        except Exception as e:
            print(f"[ADVERTENCIA] No se pudo cargar ciudad {dir_ciudad.name}: {e}")
            continue
    
    print(f"\n[INFO] Carga del dataset Kujala completada: {len(grafos)} ciudades cargadas")
    return grafos, nodos


def cargar_metros_desde_carpeta(
    raiz: Path
) -> Tuple[Dict[str, Any], Dict[str, pd.DataFrame]]:
    """Carga todas las redes desde la carpeta del dataset de 51 metros.
    
    Esta función itera sobre todos los archivos bajo la raíz e intenta
    parsear aquellos que representan grafos. Actualmente solo se soportan
    archivos JSON con listas de 'nodes' y 'edges' con nombres de campos
    interpretables. Los archivos GraphML, GEXF y GPickle se omiten porque
    requieren bibliotecas externas.
    
    Parámetros
    ----------
    raiz : Path
        Directorio raíz que contiene archivos de redes de metro.
    
    Retorna
    -------
    grafos : Dict[str, SimpleGraph]
        Diccionario con nombre de red como clave y grafo como valor.
    nodos : Dict[str, pd.DataFrame]
        Diccionario con nombre de red como clave y DataFrame de nodos como valor.
    
    Ejemplos
    --------
    >>> from pathlib import Path
    >>> grafos, nodos = cargar_metros_desde_carpeta(Path("data/metro51"))
    >>> for nombre, grafo in grafos.items():
    ...     print(f"{nombre}: {grafo.number_of_edges()} aristas")
    """
    import json
    from tarea3_module import SimpleGraph
    
    print(f"[INFO] Iniciando carga del dataset de metros desde: {raiz}")
    grafos: Dict[str, Any] = {}
    nodos: Dict[str, pd.DataFrame] = {}
    
    if not raiz.exists():
        print(f"[ADVERTENCIA] Directorio raíz no existe: {raiz}")
        return grafos, nodos
    
    print(f"[INFO] Buscando archivos de redes de metro...")
    archivos = [p for p in raiz.rglob("*") if p.is_file()]
    print(f"[INFO] Encontrados {len(archivos)} archivos para procesar")
    
    archivos_procesados = 0
    for i, ruta in enumerate(sorted(archivos), 1):
        nombre = ruta.stem
        ext = ruta.suffix.lower()
        
        print(f"\n[INFO] Procesando archivo {i}/{len(archivos)}: {ruta.name}")
        
        try:
            if ext == ".json":
                print(f"[INFO] Leyendo archivo JSON: {ruta.name}")
                with open(ruta, "r", encoding="utf-8") as f:
                    datos = json.load(f)
                
                if not isinstance(datos, dict):
                    print(f"[ADVERTENCIA] Archivo JSON no contiene un diccionario, omitiendo")
                    continue
                
                lista_nodos = datos.get("nodes") or datos.get("Nodes")
                lista_aristas = datos.get("edges") or datos.get("Edges")
                
                if not lista_nodos or not lista_aristas:
                    print(f"[ADVERTENCIA] Archivo JSON no contiene 'nodes' y 'edges', omitiendo")
                    continue
                
                print(f"[INFO] Encontrados {len(lista_nodos)} nodos y {len(lista_aristas)} aristas")
                
                # Determinar clave de ID de nodo
                nodo_muestra = lista_nodos[0]
                clave_id_nodo = None
                for candidato in ["id", "node_id", "node", "stop_id", "stop", "idNode", "key"]:
                    if candidato in nodo_muestra:
                        clave_id_nodo = candidato
                        break
                
                if clave_id_nodo is None:
                    print(f"[ADVERTENCIA] No se encontró clave de ID de nodo, omitiendo archivo")
                    continue
                
                print(f"[INFO] Usando clave '{clave_id_nodo}' como identificador de nodo")
                
                print(f"[INFO] Construyendo grafo: {nombre}")
                G = SimpleGraph(name=nombre)
                filas_atributos_nodos: List[Dict[str, Any]] = []
                
                print(f"[INFO] Agregando nodos al grafo...")
                for nd in lista_nodos:
                    nid = nd[clave_id_nodo]
                    attrs = {k: v for k, v in nd.items() if k != clave_id_nodo}
                    G.add_node(nid, **attrs)
                    fila = {"node_id": nid}
                    fila.update(attrs)
                    filas_atributos_nodos.append(fila)
                
                print(f"[INFO] {G.number_of_nodes()} nodos agregados")
                
                # Determinar claves de aristas
                arista_muestra = lista_aristas[0]
                clave_u = None
                clave_v = None
                
                for candidato in ["source", "from", "u", "from_id", "i"]:
                    if candidato in arista_muestra:
                        clave_u = candidato
                        break
                for candidato in ["target", "to", "v", "to_id", "j"]:
                    if candidato in arista_muestra:
                        clave_v = candidato
                        break
                
                if clave_u is None or clave_v is None:
                    print(f"[ADVERTENCIA] No se encontraron claves de aristas, omitiendo archivo")
                    continue
                
                print(f"[INFO] Usando claves '{clave_u}' -> '{clave_v}' para aristas")
                
                print(f"[INFO] Agregando aristas al grafo...")
                for e in lista_aristas:
                    u = e[clave_u]
                    v = e[clave_v]
                    G.add_edge(u, v)
                
                print(f"[INFO] {G.number_of_edges()} aristas agregadas")
                grafos[nombre] = G
                nodos[nombre] = pd.DataFrame(filas_atributos_nodos)
                archivos_procesados += 1
                print(f"[INFO] Red {nombre} cargada exitosamente: {G.number_of_nodes()} nodos, {G.number_of_edges()} aristas")
        
        except Exception as e:
            # Omitir archivos que no se puedan parsear
            print(f"[ADVERTENCIA] No se pudo procesar archivo {ruta.name}: {e}")
            continue
    
    print(f"\n[INFO] Carga del dataset de metros completada: {len(grafos)} redes cargadas de {archivos_procesados} archivos procesados")
    return grafos, nodos
