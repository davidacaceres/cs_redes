"""
procesar_redes
--------------

Módulo para cálculo de métricas y análisis de robustez de redes de transporte público.

Este módulo proporciona funciones para:
- Calcular métricas básicas de redes (nodos, aristas, grado promedio, etc.)
- Calcular indicadores teóricos de robustez (r_T, C_G)
- Evaluar robustez mediante simulación de remoción de nodos
- Generar resúmenes completos de datasets con procesamiento paralelo

Las funciones trabajan con la clase GrafoSimple del módulo preparar_redes
y no dependen de bibliotecas externas de grafos como networkx.

Ejemplos de uso
---------------

Calcular métricas básicas de un grafo::

    from preparar_redes import GrafoSimple
    from procesar_redes import calcular_metricas_basicas
    
    G = GrafoSimple()
    # ... agregar nodos y aristas ...
    metricas = calcular_metricas_basicas(G)
    print(f"Nodos: {metricas['n_nodos']}, Aristas: {metricas['n_aristas']}")

Generar resumen completo de un dataset::

    from procesar_redes import calcular_resumen_dataset
    
    resumen = calcular_resumen_dataset(
        grafos={"ciudad1": grafo1, "ciudad2": grafo2},
        fraccion_remover=0.2,
        ejecuciones_aleatorias=10
    )
    print(resumen)
"""

from __future__ import annotations

import random
from typing import Dict, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path, laplacian

# Importar GrafoSimple desde preparar_redes
from preparar_redes import GrafoSimple


def indicador_robustez_rT(grafo: GrafoSimple) -> float:
    """Calcula el indicador de robustez r_T para un GrafoSimple.
    
    El indicador r_T se define como (L - N + 1) / N, donde L es el número
    de enlaces y N el número de nodos. Este indicador mide la densidad de
    ciclos: valores altos indican más rutas alternativas.
    
    Parámetros
    ----------
    grafo : GrafoSimple
        El grafo a analizar.
    
    Retorna
    -------
    float
        Valor del indicador r_T. Retorna NaN si el grafo está vacío.
    
    Ejemplos
    --------
    >>> from preparar_redes import GrafoSimple
    >>> G = GrafoSimple()
    >>> G.agregar_arista(1, 2)
    >>> G.agregar_arista(2, 3)
    >>> r_T = indicador_robustez_rT(G)
    """
    n = grafo.numero_de_nodos()
    if n == 0:
        return float("nan")
    m = grafo.numero_de_aristas()
    return (m - n + 1) / n


def conductancia_efectiva_grafo_CG(grafo: GrafoSimple) -> float:
    """Calcula la conductancia efectiva del grafo C_G para un GrafoSimple.

    Para un componente conectado con matriz de adyacencia A, el Laplaciano L
    tiene valores propios λ1 ≥ λ2 ≥ ... ≥ λn=0. La resistencia efectiva del
    grafo R_G = N * Σ (1/λ_i) para i=1..n-1 y C_G = (N-1)/R_G.
    
    Cuando el grafo está desconectado, el cálculo se realiza sobre el
    componente conectado más grande. Un valor de 1 corresponde a un grafo
    perfectamente conectado (completo), mientras que valores cercanos a
    cero indican conectividad pobre.
    
    Parámetros
    ----------
    grafo : GrafoSimple
        El grafo a analizar.
    
    Retorna
    -------
    float
        Valor de conductancia efectiva C_G. Retorna NaN si el grafo está vacío.
    
    Ejemplos
    --------
    >>> from preparar_redes import GrafoSimple
    >>> G = GrafoSimple()
    >>> for i in range(5):
    ...     for j in range(i+1, 5):
    ...         G.agregar_arista(i, j)
    >>> C_G = conductancia_efectiva_grafo_CG(G)
    """
    n = grafo.numero_de_nodos()
    if n == 0:
        return float("nan")
    
    # Extraer componente conectado más grande
    comps = grafo.componentes_conectados()
    if not comps:
        return float("nan")
    mas_grande = max(comps, key=len)
    if len(mas_grande) < 2:
        return 0.0
    
    # Construir matriz de adyacencia para el componente
    lista_nodos = list(mas_grande)
    mapa_idx = {nodo: idx for idx, nodo in enumerate(lista_nodos)}
    filas = []
    cols = []
    for u in lista_nodos:
        for v in grafo._adj[u]:
            if v in mapa_idx:
                filas.append(mapa_idx[u])
                cols.append(mapa_idx[v])
    datos = np.ones(len(filas), dtype=float)
    n_lcc = len(lista_nodos)
    adj = csr_matrix((datos, (filas, cols)), shape=(n_lcc, n_lcc))
    
    # Calcular valores propios del Laplaciano
    L = laplacian(adj, normed=False)
    # Convertir a denso para valores propios; para grafos moderados esto es aceptable
    L_denso = L.toarray()
    eigvals = np.linalg.eigvalsh(L_denso)
    
    # Remover valores propios cero (tolerancia)
    no_cero = [lam for lam in eigvals if lam > 1e-9]
    if not no_cero:
        return 0.0
    R_G = n_lcc * float(np.sum(1.0 / np.array(no_cero)))
    return float((n_lcc - 1) / R_G)


def calcular_metricas_basicas(grafo: GrafoSimple) -> Dict[str, Union[float, int]]:
    """Calcula métricas básicas de red para un GrafoSimple.
    
    Calcula:
    - Número de nodos
    - Número de aristas
    - Grado promedio
    - Fracción del componente gigante
    - Longitud promedio de camino más corto
    - Coeficiente de clustering promedio
    
    Parámetros
    ----------
    grafo : GrafoSimple
        El grafo a analizar.
    
    Retorna
    -------
    Dict[str, Union[float, int]]
        Diccionario con las métricas calculadas.
    
    Ejemplos
    --------
    >>> from preparar_redes import GrafoSimple
    >>> G = GrafoSimple()
    >>> G.agregar_arista(1, 2)
    >>> G.agregar_arista(2, 3)
    >>> metricas = calcular_metricas_basicas(G)
    >>> print(metricas['n_nodos'])
    3
    """
    n = grafo.numero_de_nodos()
    m = grafo.numero_de_aristas()
    metricas = {
        "n_nodos": n,
        "n_aristas": m,
        "grado_promedio": float(2 * m / n) if n > 0 else float("nan"),
        "fraccion_gigante": float("nan"),
        "longitud_camino_promedio": float("nan"),
        "clustering_promedio": float("nan"),
    }
    
    if n == 0:
        return metricas
    
    comps = grafo.componentes_conectados()
    if not comps:
        metricas["fraccion_gigante"] = float("nan")
    else:
        mas_grande = max(comps, key=len)
        metricas["fraccion_gigante"] = len(mas_grande) / n
        
        # Calcular longitud promedio de camino más corto en el componente más grande
        if len(mas_grande) > 1:
            # Construir adyacencia para este componente
            lista_nodos = list(mas_grande)
            mapa_idx = {nodo: idx for idx, nodo in enumerate(lista_nodos)}
            filas = []
            cols = []
            for u in lista_nodos:
                for v in grafo._adj[u]:
                    if v in mapa_idx:
                        filas.append(mapa_idx[u])
                        cols.append(mapa_idx[v])
            datos = np.ones(len(filas), dtype=float)
            n_lcc = len(lista_nodos)
            adj = csr_matrix((datos, (filas, cols)), shape=(n_lcc, n_lcc))
            matriz_dist = shortest_path(adj, method='D', directed=False, unweighted=True)
            
            # Solo distancias finitas
            dists_finitas = matriz_dist[np.isfinite(matriz_dist)]
            if len(dists_finitas) > 1:
                # Excluir distancias de longitud cero ignorando la diagonal
                dists_finitas = dists_finitas[dists_finitas > 0]
                metricas["longitud_camino_promedio"] = float(np.mean(dists_finitas))
    
    # Coeficiente de clustering
    # Para cada nodo, contar triángulos y triples posibles
    valores_clustering = []
    for u in grafo.nodos():
        vecinos = grafo._adj[u]
        k = len(vecinos)
        if k < 2:
            continue
        # Contar pares de vecinos que están conectados
        tri = 0
        lista_vecinos = list(vecinos)
        for i in range(k):
            for j in range(i + 1, k):
                v = lista_vecinos[i]
                w = lista_vecinos[j]
                if grafo.tiene_arista(v, w):
                    tri += 1
        valores_clustering.append(tri / (k * (k - 1) / 2))
    
    if valores_clustering:
        metricas["clustering_promedio"] = float(np.mean(valores_clustering))
    else:
        metricas["clustering_promedio"] = 0.0
    
    return metricas


def indice_robustez_simple(
    grafo: GrafoSimple,
    fraccion_remover: float = 0.2,
    estrategia: str = "grado",
    semilla: Optional[int] = None,
) -> float:
    """Calcula un índice de robustez simple removiendo una fracción de nodos.

    Parámetros
    ----------
    grafo : GrafoSimple
        El grafo a analizar.
    fraccion_remover : float
        Fracción de nodos a remover (0 < frac <= 1).
    estrategia : str
        Estrategia de remoción: "grado" remueve los nodos de mayor grado;
        "aleatorio" remueve uniformemente al azar.
    semilla : int o None
        Semilla aleatoria para reproducibilidad en el caso aleatorio.

    Retorna
    -------
    float
        Fracción de nodos en el componente conectado más grande después
        de la remoción relativa al número original de nodos.
    
    Ejemplos
    --------
    >>> from preparar_redes import GrafoSimple
    >>> G = GrafoSimple()
    >>> for i in range(10):
    ...     G.agregar_arista(i, (i+1) % 10)
    >>> robustez = indice_robustez_simple(G, fraccion_remover=0.2, estrategia="grado")
    """
    n = grafo.numero_de_nodos()
    if n == 0:
        return float("nan")
    
    n_remover = max(1, int(fraccion_remover * n))
    if n_remover >= n:
        n_remover = n - 1
    
    if estrategia == "grado":
        grados = grafo.grado()
        # Ordenar por grado descendente; en empate, por id de nodo para asegurar determinismo
        nodos_ordenados = sorted(grados.items(), key=lambda x: (-x[1], x[0]))
        a_remover = [nid for nid, _ in nodos_ordenados[:n_remover]]
    elif estrategia == "aleatorio":
        rng = random.Random(semilla)
        a_remover = rng.sample(grafo.nodos(), n_remover)
    else:
        raise ValueError("estrategia debe ser 'grado' o 'aleatorio'")
    
    H = grafo.copiar()
    H.remover_nodos(a_remover)
    
    if H.numero_de_nodos() == 0:
        return 0.0
    
    comps = H.componentes_conectados()
    if not comps:
        return 0.0
    
    mas_grande = max(comps, key=len)
    return len(mas_grande) / n


def _trabajador_calcular_metricas(argumentos: Tuple[str, GrafoSimple, float, int, Optional[int]]) -> Dict[str, Union[str, float, int]]:
    """Worker function para cálculo paralelo de métricas.
    
    Parámetros
    ----------
    argumentos : Tuple[str, GrafoSimple, float, int, Optional[int]]
        Tupla con (nombre, grafo, fraccion_remover, ejecuciones_aleatorias, semilla).
        
    Retorna
    -------
    Dict[str, Union[str, float, int]]
        Diccionario con todas las métricas calculadas para el grafo.
    """
    nombre, G, fraccion_remover, ejecuciones_aleatorias, semilla = argumentos
    
    print(f"[INFO] Calculando métricas para: {nombre}")
    
    metricas = calcular_metricas_basicas(G)
    r_T = indicador_robustez_rT(G)
    C_G = conductancia_efectiva_grafo_CG(G)
    rob_grado = indice_robustez_simple(G, fraccion_remover=fraccion_remover, estrategia="grado")
    
    # Promediar remociones aleatorias
    rob_aleatorio_total = 0.0
    for i in range(max(1, ejecuciones_aleatorias)):
        semilla_ejecucion = None if semilla is None else semilla + i
        rob_aleatorio_total += indice_robustez_simple(
            G, fraccion_remover=fraccion_remover, estrategia="aleatorio", semilla=semilla_ejecucion
        )
    rob_aleatorio = rob_aleatorio_total / max(1, ejecuciones_aleatorias)
    
    fila = {
        "nombre": nombre,
        **metricas,
        "r_T": r_T,
        "C_G": C_G,
        f"robustez_grado_{int(fraccion_remover*100)}pct": rob_grado,
        f"robustez_aleatorio_{int(fraccion_remover*100)}pct": rob_aleatorio,
    }
    
    print(f"[INFO] Métricas calculadas para: {nombre}")
    return fila


def calcular_resumen_dataset(
    grafos: Dict[str, GrafoSimple],
    fraccion_remover: float = 0.2,
    ejecuciones_aleatorias: int = 10,
    semilla: Optional[int] = None,
    usar_paralelo: bool = True,
    max_workers: Optional[int] = None
) -> pd.DataFrame:
    """Calcula un resumen DataFrame para un conjunto de grafos.
    
    Para cada grafo, calcula métricas básicas, indicadores de robustez teóricos
    (r_T, C_G) y métricas de robustez mediante simulación de remoción de nodos.
    Usa procesamiento paralelo con threading para mejorar el rendimiento.
    
    Parámetros
    ----------
    grafos : Dict[str, GrafoSimple]
        Diccionario de grafos a analizar.
    fraccion_remover : float, opcional
        Fracción de nodos a remover en simulaciones (por defecto 0.2).
    ejecuciones_aleatorias : int, opcional
        Número de ejecuciones para remoción aleatoria (por defecto 10).
    semilla : int o None, opcional
        Semilla aleatoria para reproducibilidad.
    usar_paralelo : bool, opcional
        Si True, usa ThreadPoolExecutor para procesamiento paralelo (por defecto True).
    max_workers : int, opcional
        Número máximo de workers. Si es None, usa el número de CPUs disponibles.
    
    Retorna
    -------
    pd.DataFrame
        DataFrame con una fila por grafo y columnas para todas las métricas.
    
    Ejemplos
    --------
    >>> from preparar_redes import GrafoSimple
    >>> grafos = {"red1": grafo1, "red2": grafo2}
    >>> resumen = calcular_resumen_dataset(grafos, fraccion_remover=0.2)
    >>> print(resumen[['nombre', 'n_nodos', 'n_aristas', 'r_T']])
    """
    print(f"[INFO] Iniciando cálculo de resumen para {len(grafos)} grafos")
    
    if not usar_paralelo or len(grafos) <= 1:
        # Procesamiento secuencial
        filas = []
        for nombre, G in grafos.items():
            fila = _trabajador_calcular_metricas((nombre, G, fraccion_remover, ejecuciones_aleatorias, semilla))
            filas.append(fila)
    else:
        # Procesamiento paralelo
        if max_workers is None:
            max_workers = min(multiprocessing.cpu_count(), len(grafos))
        
        print(f"[INFO] Usando procesamiento paralelo con {max_workers} workers")
        
        # Preparar argumentos para workers
        tareas = [(nombre, G, fraccion_remover, ejecuciones_aleatorias, semilla) 
                  for nombre, G in grafos.items()]
        
        filas = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Enviar todas las tareas
            futuros = {executor.submit(_trabajador_calcular_metricas, tarea): tarea[0] 
                      for tarea in tareas}
            
            # Procesar resultados a medida que se completan
            for futuro in as_completed(futuros):
                fila = futuro.result()
                filas.append(fila)
    
    print(f"[INFO] Cálculo de resumen completado para {len(filas)} grafos")
    return pd.DataFrame(filas)
