"""
preparar_redes
--------------

Módulo para construcción de grafos desde datos de redes de transporte público.

Este módulo proporciona:
- Clase GrafoSimple: implementación de grafo no dirigido con conjuntos de adyacencia
- Construcción de grafos desde datasets Kujala (formato CSV)
- Construcción de grafos desde datasets Metro51 (formato JSON)
- Procesamiento paralelo con threading para mejorar rendimiento

La clase GrafoSimple no depende de bibliotecas externas de grafos como networkx,
usando solo numpy y scipy para operaciones matriciales.

Ejemplos de uso
---------------

Cargar dataset Kujala con procesamiento paralelo::

    from pathlib import Path
    from preparar_redes import cargar_dataset_kujala
    
    grafos, nodos = cargar_dataset_kujala(Path("data/kujala/procesado"))
    for ciudad, grafo in grafos.items():
        print(f"{ciudad}: {grafo.numero_de_nodos()} nodos")

Construir un grafo individual::

    from preparar_redes import construir_grafo_desde_ciudad_kujala
    
    grafo, nodos_df = construir_grafo_desde_ciudad_kujala(
        Path("data/kujala/procesado/adelaide")
    )
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


class GrafoSimple:
    """Un grafo no dirigido simple implementado con conjuntos de adyacencia.

    Esta clase proporciona una interfaz mínima para trabajar con grafos sin
    depender de bibliotecas externas como networkx. Soporta agregar nodos y
    aristas, consultar el número de nodos y aristas, inspeccionar grados,
    copiar, remover nodos e iterar sobre nodos y aristas.
    
    Atributos
    ---------
    nombre : str
        Nombre descriptivo del grafo.
    _adj : Dict[Any, Set[Any]]
        Diccionario de conjuntos de adyacencia.
    atributos_nodos : Dict[Any, Dict[str, Any]]
        Diccionario de atributos por nodo.
    """

    def __init__(self, nombre: str = "") -> None:
        """Inicializa un grafo vacío.
        
        Parámetros
        ----------
        nombre : str, opcional
            Nombre descriptivo del grafo.
        """
        self.nombre = nombre
        self._adj: Dict[Any, Set[Any]] = {}
        self.atributos_nodos: Dict[Any, Dict[str, Any]] = {}

    def agregar_nodo(self, id_nodo: Any, **atributos: Any) -> None:
        """Agrega un nodo al grafo, opcionalmente con atributos.
        
        Parámetros
        ----------
        id_nodo : Any
            Identificador del nodo.
        **atributos : Any
            Atributos opcionales del nodo (ej: lat=40.7, lon=-74.0).
        """
        if id_nodo not in self._adj:
            self._adj[id_nodo] = set()
        if id_nodo not in self.atributos_nodos:
            self.atributos_nodos[id_nodo] = {}
        # Actualizar atributos
        self.atributos_nodos[id_nodo].update(atributos)

    def agregar_arista(self, u: Any, v: Any) -> None:
        """Agrega una arista no dirigida entre u y v.
        
        Parámetros
        ----------
        u : Any
            Primer nodo de la arista.
        v : Any
            Segundo nodo de la arista.
        """
        if u == v:
            # Los auto-bucles se ignoran
            return
        self._adj.setdefault(u, set()).add(v)
        self._adj.setdefault(v, set()).add(u)

    def tiene_arista(self, u: Any, v: Any) -> bool:
        """Verifica si existe una arista entre u y v.
        
        Parámetros
        ----------
        u : Any
            Primer nodo.
        v : Any
            Segundo nodo.
            
        Retorna
        -------
        bool
            True si existe la arista, False en caso contrario.
        """
        return v in self._adj.get(u, set())

    def nodos(self) -> List[Any]:
        """Retorna una lista de todos los nodos del grafo.
        
        Retorna
        -------
        List[Any]
            Lista de identificadores de nodos.
        """
        return list(self._adj.keys())

    def aristas(self) -> List[Tuple[Any, Any]]:
        """Retorna una lista de todas las aristas del grafo.
        
        Cada arista se representa como una tupla (u, v) donde u <= v.
        
        Retorna
        -------
        List[Tuple[Any, Any]]
            Lista de aristas.
        """
        ed = []
        for u, nbrs in self._adj.items():
            for v in nbrs:
                if u <= v:
                    ed.append((u, v))
        return ed

    def numero_de_nodos(self) -> int:
        """Retorna el número de nodos en el grafo.
        
        Retorna
        -------
        int
            Cantidad de nodos.
        """
        return len(self._adj)

    def numero_de_aristas(self) -> int:
        """Retorna el número de aristas en el grafo.
        
        Retorna
        -------
        int
            Cantidad de aristas.
        """
        return sum(len(nbrs) for nbrs in self._adj.values()) // 2

    def grado(self, nodo: Optional[Any] = None) -> Union[int, Dict[Any, int]]:
        """Retorna el grado de un nodo o un mapeo de grados.
        
        Parámetros
        ----------
        nodo : Any, opcional
            Si se especifica, retorna el grado de ese nodo.
            Si es None, retorna un diccionario con todos los grados.
            
        Retorna
        -------
        int o Dict[Any, int]
            Grado del nodo o diccionario de grados.
        """
        if nodo is not None:
            return len(self._adj.get(nodo, set()))
        else:
            return {n: len(nbrs) for n, nbrs in self._adj.items()}

    def copiar(self) -> "GrafoSimple":
        """Retorna una copia profunda del grafo.
        
        Retorna
        -------
        GrafoSimple
            Copia del grafo.
        """
        nuevo_g = GrafoSimple(nombre=self.nombre)
        # Copiar adyacencia
        for u, nbrs in self._adj.items():
            nuevo_g._adj[u] = set(nbrs)
        # Copiar atributos
        for n, attrs in self.atributos_nodos.items():
            nuevo_g.atributos_nodos[n] = dict(attrs)
        return nuevo_g

    def remover_nodos(self, nodos: Iterable[Any]) -> None:
        """Remueve múltiples nodos del grafo.
        
        Parámetros
        ----------
        nodos : Iterable[Any]
            Nodos a remover.
        """
        for n in nodos:
            if n in self._adj:
                for nbr in list(self._adj[n]):
                    self._adj[nbr].discard(n)
                del self._adj[n]
                self.atributos_nodos.pop(n, None)

    def _a_matriz_dispersa(
        self, orden_nodos: Optional[List[Any]] = None
    ) -> Tuple[csr_matrix, Dict[Any, int], List[Any]]:
        """Convierte el grafo en una matriz de adyacencia dispersa de SciPy.

        Retorna la matriz CSR, un mapeo de node_id a índice, y el orden de nodos.
        
        Parámetros
        ----------
        orden_nodos : List[Any], opcional
            Orden específico de nodos. Si es None, usa el orden de self.nodos().
            
        Retorna
        -------
        Tuple[csr_matrix, Dict[Any, int], List[Any]]
            Matriz CSR, mapeo de índices, orden de nodos.
        """
        if orden_nodos is None:
            orden_nodos = self.nodos()
        mapa_indices = {nodo: idx for idx, nodo in enumerate(orden_nodos)}
        n = len(orden_nodos)
        filas: List[int] = []
        cols: List[int] = []
        for u, nbrs in self._adj.items():
            if u not in mapa_indices:
                continue
            u_idx = mapa_indices[u]
            for v in nbrs:
                if v not in mapa_indices:
                    continue
                v_idx = mapa_indices[v]
                filas.append(u_idx)
                cols.append(v_idx)
        datos = np.ones(len(filas), dtype=float)
        adj = csr_matrix((datos, (filas, cols)), shape=(n, n))
        return adj, mapa_indices, orden_nodos

    def esta_conectado(self) -> bool:
        """Retorna True si el grafo está conectado, False en caso contrario.
        
        Retorna
        -------
        bool
            True si el grafo es conexo.
        """
        if self.numero_de_nodos() == 0:
            return True
        adj, _, _ = self._a_matriz_dispersa()
        n_componentes, etiquetas = connected_components(adj, directed=False)
        return n_componentes == 1

    def componentes_conectados(self) -> List[Set[Any]]:
        """Retorna una lista de conjuntos de nodos, uno por componente conectado.
        
        Retorna
        -------
        List[Set[Any]]
            Lista de componentes conectados.
        """
        if self.numero_de_nodos() == 0:
            return []
        lista_nodos = self.nodos()
        adj, mapa_indices, orden = self._a_matriz_dispersa(orden_nodos=lista_nodos)
        n_componentes, etiquetas = connected_components(adj, directed=False)
        comps: List[Set[Any]] = [set() for _ in range(n_componentes)]
        for nodo, lbl in zip(lista_nodos, etiquetas):
            comps[lbl].add(nodo)
        return comps


def construir_grafo_desde_ciudad_kujala(
    directorio_ciudad: Path
) -> Tuple[GrafoSimple, pd.DataFrame]:
    """Construye un GrafoSimple desde un directorio de ciudad del dataset Kujala.
    
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
    grafo : GrafoSimple
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
    >>> print(f"Nodos: {grafo.numero_de_nodos()}, Aristas: {grafo.numero_de_aristas()}")
    """
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
    G = GrafoSimple(nombre=f"kujala_{directorio_ciudad.name}")
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
        G.agregar_nodo(nid, **attrs)
        atributos_nodos.append({"node_id": nid, **attrs})
    
    print(f"[INFO] {G.numero_de_nodos()} nodos agregados al grafo")
    
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
        G.agregar_arista(u_int, v_int)
        aristas_agregadas += 1
    
    print(f"[INFO] {aristas_agregadas} aristas agregadas al grafo")
    
    nodos_df_salida = pd.DataFrame(atributos_nodos)
    print(f"[INFO] Grafo construido exitosamente para {directorio_ciudad.name}: {G.numero_de_nodos()} nodos, {G.numero_de_aristas()} aristas")
    return G, nodos_df_salida


def _trabajador_construir_grafo_kujala(argumentos: Tuple[Path, int, int]) -> Tuple[str, Optional[GrafoSimple], Optional[pd.DataFrame], Optional[str]]:
    """Worker function para construcción paralela de grafos Kujala.
    
    Parámetros
    ----------
    argumentos : Tuple[Path, int, int]
        Tupla con (directorio_ciudad, índice, total).
        
    Retorna
    -------
    Tuple[str, Optional[GrafoSimple], Optional[pd.DataFrame], Optional[str]]
        Tupla con (nombre_ciudad, grafo, nodos_df, mensaje_error).
    """
    dir_ciudad, idx, total = argumentos
    try:
        G, df_nodos = construir_grafo_desde_ciudad_kujala(dir_ciudad)
        return dir_ciudad.name, G, df_nodos, None
    except Exception as e:
        error_msg = f"[ADVERTENCIA] No se pudo cargar ciudad {dir_ciudad.name}: {e}"
        print(error_msg)
        return dir_ciudad.name, None, None, str(e)


def cargar_dataset_kujala(
    raiz: Path,
    usar_paralelo: bool = True,
    max_workers: Optional[int] = None
) -> Tuple[Dict[str, GrafoSimple], Dict[str, pd.DataFrame]]:
    """Carga todas las redes de ciudades desde el directorio del dataset Kujala.
    
    Itera sobre todos los subdirectorios en la ruta raíz e intenta construir
    un grafo para cada ciudad encontrada. Las ciudades que no puedan ser
    procesadas se omiten. Usa procesamiento paralelo con threading para
    mejorar el rendimiento.
    
    Parámetros
    ----------
    raiz : Path
        Directorio raíz que contiene subdirectorios de ciudades.
    usar_paralelo : bool, opcional
        Si True, usa ThreadPoolExecutor para procesamiento paralelo (por defecto True).
    max_workers : int, opcional
        Número máximo de workers. Si es None, usa el número de CPUs disponibles.
    
    Retorna
    -------
    grafos : Dict[str, GrafoSimple]
        Diccionario con nombre de ciudad como clave y grafo como valor.
    nodos : Dict[str, pd.DataFrame]
        Diccionario con nombre de ciudad como clave y DataFrame de nodos como valor.
    
    Ejemplos
    --------
    >>> from pathlib import Path
    >>> grafos, nodos = cargar_dataset_kujala(Path("data/kujala"))
    >>> for ciudad, grafo in grafos.items():
    ...     print(f"{ciudad}: {grafo.numero_de_nodos()} nodos")
    """
    print(f"[INFO] Iniciando carga del dataset Kujala desde: {raiz}")
    grafos: Dict[str, GrafoSimple] = {}
    nodos: Dict[str, pd.DataFrame] = {}
    
    if not raiz.exists():
        print(f"[ADVERTENCIA] Directorio raíz no existe: {raiz}")
        return grafos, nodos
    
    directorios_ciudad = sorted([p for p in raiz.iterdir() if p.is_dir()])
    print(f"[INFO] Encontrados {len(directorios_ciudad)} directorios de ciudades")
    
    if not usar_paralelo or len(directorios_ciudad) <= 1:
        # Procesamiento secuencial
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
    else:
        # Procesamiento paralelo
        if max_workers is None:
            max_workers = min(multiprocessing.cpu_count(), len(directorios_ciudad))
        
        print(f"[INFO] Usando procesamiento paralelo con {max_workers} workers")
        
        # Preparar argumentos para workers
        tareas = [(dir_ciudad, i, len(directorios_ciudad)) 
                  for i, dir_ciudad in enumerate(directorios_ciudad, 1)]
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Enviar todas las tareas
            futuros = {executor.submit(_trabajador_construir_grafo_kujala, tarea): tarea[0] 
                      for tarea in tareas}
            
            # Procesar resultados a medida que se completan
            for futuro in as_completed(futuros):
                nombre_ciudad, G, df_nodos, error = futuro.result()
                if G is not None and df_nodos is not None:
                    grafos[nombre_ciudad] = G
                    nodos[nombre_ciudad] = df_nodos
                    print(f"[INFO] Ciudad {nombre_ciudad} cargada exitosamente")
    
    print(f"\n[INFO] Carga del dataset Kujala completada: {len(grafos)} ciudades cargadas")
    return grafos, nodos


def _trabajador_construir_grafo_metro(argumentos: Tuple[Path, int, int]) -> Tuple[str, Optional[GrafoSimple], Optional[pd.DataFrame], Optional[str]]:
    """Worker function para construcción paralela de grafos Metro51.
    
    Parámetros
    ----------
    argumentos : Tuple[Path, int, int]
        Tupla con (ruta_archivo, índice, total).
        
    Retorna
    -------
    Tuple[str, Optional[GrafoSimple], Optional[pd.DataFrame], Optional[str]]
        Tupla con (nombre_red, grafo, nodos_df, mensaje_error).
    """
    ruta, idx, total = argumentos
    nombre = ruta.stem
    ext = ruta.suffix.lower()
    
    try:
        if ext != ".json":
            return nombre, None, None, "No es archivo JSON"
        
        with open(ruta, "r", encoding="utf-8") as f:
            datos = json.load(f)
        
        if not isinstance(datos, dict):
            return nombre, None, None, "JSON no contiene un diccionario"
        
        lista_nodos = datos.get("nodes") or datos.get("Nodes")
        lista_aristas = datos.get("edges") or datos.get("Edges")
        
        if not lista_nodos or not lista_aristas:
            return nombre, None, None, "JSON no contiene 'nodes' y 'edges'"
        
        # Determinar clave de ID de nodo
        nodo_muestra = lista_nodos[0]
        clave_id_nodo = None
        for candidato in ["id", "node_id", "node", "stop_id", "stop", "idNode", "key"]:
            if candidato in nodo_muestra:
                clave_id_nodo = candidato
                break
        
        if clave_id_nodo is None:
            return nombre, None, None, "No se encontró clave de ID de nodo"
        
        G = GrafoSimple(nombre=nombre)
        filas_atributos_nodos: List[Dict[str, Any]] = []
        
        for nd in lista_nodos:
            nid = nd[clave_id_nodo]
            attrs = {k: v for k, v in nd.items() if k != clave_id_nodo}
            G.agregar_nodo(nid, **attrs)
            fila = {"node_id": nid}
            fila.update(attrs)
            filas_atributos_nodos.append(fila)
        
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
            return nombre, None, None, "No se encontraron claves de aristas"
        
        for e in lista_aristas:
            u = e[clave_u]
            v = e[clave_v]
            G.agregar_arista(u, v)
        
        return nombre, G, pd.DataFrame(filas_atributos_nodos), None
        
    except Exception as e:
        return nombre, None, None, str(e)


def cargar_metros_desde_carpeta(
    raiz: Path,
    usar_paralelo: bool = True,
    max_workers: Optional[int] = None
) -> Tuple[Dict[str, GrafoSimple], Dict[str, pd.DataFrame]]:
    """Carga todas las redes desde la carpeta del dataset de 51 metros.
    
    Esta función itera sobre todos los archivos bajo la raíz e intenta
    parsear aquellos que representan grafos. Actualmente solo se soportan
    archivos JSON con listas de 'nodes' y 'edges'. Usa procesamiento
    paralelo con threading para mejorar el rendimiento.
    
    Parámetros
    ----------
    raiz : Path
        Directorio raíz que contiene archivos de redes de metro.
    usar_paralelo : bool, opcional
        Si True, usa ThreadPoolExecutor para procesamiento paralelo (por defecto True).
    max_workers : int, opcional
        Número máximo de workers. Si es None, usa el número de CPUs disponibles.
    
    Retorna
    -------
    grafos : Dict[str, GrafoSimple]
        Diccionario con nombre de red como clave y grafo como valor.
    nodos : Dict[str, pd.DataFrame]
        Diccionario con nombre de red como clave y DataFrame de nodos como valor.
    
    Ejemplos
    --------
    >>> from pathlib import Path
    >>> grafos, nodos = cargar_metros_desde_carpeta(Path("data/metro51"))
    >>> for nombre, grafo in grafos.items():
    ...     print(f"{nombre}: {grafo.numero_de_aristas()} aristas")
    """
    print(f"[INFO] Iniciando carga del dataset de metros desde: {raiz}")
    grafos: Dict[str, GrafoSimple] = {}
    nodos: Dict[str, pd.DataFrame] = {}
    
    if not raiz.exists():
        print(f"[ADVERTENCIA] Directorio raíz no existe: {raiz}")
        return grafos, nodos
    
    print(f"[INFO] Buscando archivos de redes de metro...")
    archivos = [p for p in raiz.rglob("*") if p.is_file() and p.suffix.lower() == ".json"]
    print(f"[INFO] Encontrados {len(archivos)} archivos JSON para procesar")
    
    if not usar_paralelo or len(archivos) <= 1:
        # Procesamiento secuencial
        for i, ruta in enumerate(sorted(archivos), 1):
            print(f"\n[INFO] Procesando archivo {i}/{len(archivos)}: {ruta.name}")
            nombre, G, df_nodos, error = _trabajador_construir_grafo_metro((ruta, i, len(archivos)))
            if G is not None and df_nodos is not None:
                grafos[nombre] = G
                nodos[nombre] = df_nodos
                print(f"[INFO] Red {nombre} cargada: {G.numero_de_nodos()} nodos, {G.numero_de_aristas()} aristas")
            else:
                print(f"[ADVERTENCIA] No se pudo procesar {ruta.name}: {error}")
    else:
        # Procesamiento paralelo
        if max_workers is None:
            max_workers = min(multiprocessing.cpu_count(), len(archivos))
        
        print(f"[INFO] Usando procesamiento paralelo con {max_workers} workers")
        
        # Preparar argumentos para workers
        tareas = [(ruta, i, len(archivos)) for i, ruta in enumerate(sorted(archivos), 1)]
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Enviar todas las tareas
            futuros = {executor.submit(_trabajador_construir_grafo_metro, tarea): tarea[0] 
                      for tarea in tareas}
            
            # Procesar resultados a medida que se completan
            for futuro in as_completed(futuros):
                nombre, G, df_nodos, error = futuro.result()
                if G is not None and df_nodos is not None:
                    grafos[nombre] = G
                    nodos[nombre] = df_nodos
                    print(f"[INFO] Red {nombre} cargada: {G.numero_de_nodos()} nodos, {G.numero_de_aristas()} aristas")
    
    print(f"\n[INFO] Carga del dataset de metros completada: {len(grafos)} redes cargadas")
    return grafos, nodos
