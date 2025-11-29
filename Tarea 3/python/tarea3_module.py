"""
tarea3_module
-------------------

Módulo principal para análisis de redes de transporte público sin depender
de bibliotecas externas de grafos.

La clase central en este módulo es :class:`SimpleGraph`, una estructura
de grafo mínima implementada usando conjuntos de adyacencia. Soporta
operaciones básicas como agregar nodos y aristas, copiar, eliminar nodos
y calcular grados.

Este módulo proporciona funciones para:
- Construir grafos desde formatos comunes (CSV, JSON)
- Calcular métricas de robustez (índice ciclomático r_T, conductancia C_G)
- Calcular métricas estructurales básicas
- Evaluar robustez mediante remoción de nodos

Para funciones de descarga y preparación de datasets, ver el módulo
:mod:`dataset_loader`.

Estas funciones evitan dependencias en la biblioteca ``networkx``,
que puede no estar disponible en algunos entornos de ejecución. En su lugar
usan ``numpy`` y ``scipy.sparse.csgraph`` cuando es apropiado.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union, Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path, laplacian

# Importar funciones de descarga y preparación de datasets desde el módulo dedicado
# Mantener nombres originales en inglés para compatibilidad con código existente
from dataset_loader import (
    descargar_archivo as download_file,
    extraer_zip as extract_zip,
    construir_grafo_desde_ciudad_kujala as build_graph_from_kujala_city,
    cargar_dataset_kujala as load_kujala_dataset,
    cargar_metros_desde_carpeta as load_metro51_from_folder,
)


class SimpleGraph:
    """A simple undirected graph implemented with adjacency sets.

    This class provides a minimal interface required by the assignment
    functions without depending on external graph libraries such as
    ``networkx``.  It supports adding nodes and edges, querying the
    number of nodes and edges, inspecting degrees, copying, removal of
    nodes, and iterating over nodes and edges.
    """

    def __init__(self, name: str = "") -> None:
        self.name = name
        self._adj: Dict[Any, Set[Any]] = {}
        self.node_attrs: Dict[Any, Dict[str, Any]] = {}

    def add_node(self, node_id: Any, **attrs: Any) -> None:
        """Add a node to the graph, optionally with attributes."""
        if node_id not in self._adj:
            self._adj[node_id] = set()
        if node_id not in self.node_attrs:
            self.node_attrs[node_id] = {}
        # Update attributes
        self.node_attrs[node_id].update(attrs)

    def add_edge(self, u: Any, v: Any) -> None:
        """Add an undirected edge between ``u`` and ``v``."""
        if u == v:
            # Self loops are ignored
            return
        self._adj.setdefault(u, set()).add(v)
        self._adj.setdefault(v, set()).add(u)

    def has_edge(self, u: Any, v: Any) -> bool:
        return v in self._adj.get(u, set())

    def nodes(self) -> List[Any]:
        return list(self._adj.keys())

    def edges(self) -> List[Tuple[Any, Any]]:
        ed = []
        for u, nbrs in self._adj.items():
            for v in nbrs:
                if u <= v:
                    ed.append((u, v))
        return ed

    def number_of_nodes(self) -> int:
        return len(self._adj)

    def number_of_edges(self) -> int:
        return sum(len(nbrs) for nbrs in self._adj.values()) // 2

    def degree(self, node: Optional[Any] = None) -> Union[int, Dict[Any, int]]:
        """Return degree of a node or mapping of degrees."""
        if node is not None:
            return len(self._adj.get(node, set()))
        else:
            return {n: len(nbrs) for n, nbrs in self._adj.items()}

    def copy(self) -> "SimpleGraph":
        """Return a deep copy of the graph."""
        new_g = SimpleGraph(name=self.name)
        # Copy adjacency
        for u, nbrs in self._adj.items():
            new_g._adj[u] = set(nbrs)
        # Copy attributes
        for n, attrs in self.node_attrs.items():
            new_g.node_attrs[n] = dict(attrs)
        return new_g

    def remove_nodes_from(self, nodes: Iterable[Any]) -> None:
        """Remove multiple nodes from the graph."""
        for n in nodes:
            if n in self._adj:
                for nbr in list(self._adj[n]):
                    self._adj[nbr].discard(n)
                del self._adj[n]
                self.node_attrs.pop(n, None)

    # Additional helpers for connected components and path lengths
    def _to_sparse_matrix(self, node_order: Optional[List[Any]] = None) -> Tuple[csr_matrix, Dict[Any, int], List[Any]]:
        """Convert the graph into a SciPy sparse adjacency matrix.

        Returns the CSR matrix, a mapping from node_id to index, and the node order.
        """
        if node_order is None:
            node_order = self.nodes()
        index_map = {node: idx for idx, node in enumerate(node_order)}
        n = len(node_order)
        rows: List[int] = []
        cols: List[int] = []
        for u, nbrs in self._adj.items():
            if u not in index_map:
                continue
            u_idx = index_map[u]
            for v in nbrs:
                if v not in index_map:
                    continue
                v_idx = index_map[v]
                rows.append(u_idx)
                cols.append(v_idx)
        data = np.ones(len(rows), dtype=float)
        adj = csr_matrix((data, (rows, cols)), shape=(n, n))
        return adj, index_map, node_order

    def is_connected(self) -> bool:
        """Return True if the graph is connected, False otherwise."""
        if self.number_of_nodes() == 0:
            return True
        adj, _, _ = self._to_sparse_matrix()
        n_components, labels = connected_components(adj, directed=False)
        return n_components == 1

    def connected_components(self) -> List[Set[Any]]:
        """Return a list of sets of nodes, one per connected component."""
        if self.number_of_nodes() == 0:
            return []
        nodes_list = self.nodes()
        adj, index_map, order = self._to_sparse_matrix(node_order=nodes_list)
        n_components, labels = connected_components(adj, directed=False)
        comps: List[Set[Any]] = [set() for _ in range(n_components)]
        for node, lbl in zip(nodes_list, labels):
            comps[lbl].add(node)
        return comps


# Las funciones de descarga y preparación de datasets han sido movidas al módulo dataset_loader.
# Se mantienen las importaciones con nombres en inglés arriba para compatibilidad con código existente.


def robustness_indicator_rT(G: SimpleGraph) -> float:
    """Compute the robustness indicator r_T for a SimpleGraph."""
    n = G.number_of_nodes()
    if n == 0:
        return float("nan")
    m = G.number_of_edges()
    return (m - n + 1) / n


def effective_graph_conductance_CG(G: SimpleGraph) -> float:
    """Compute the effective graph conductance C_G for a SimpleGraph.

    For a connected component with adjacency matrix A, the Laplacian L
    has eigenvalues λ1 ≥ λ2 ≥ ... ≥ λn=0.  The effective graph
    resistance R_G = N * Σ (1/λ_i) for i=1..n-1 and C_G = (N-1)/R_G.
    When the graph is disconnected, the computation is performed on
    the largest connected component.  A value of 1 corresponds to a
    perfectly connected graph (complete), while values approaching
    zero indicate poor connectivity.
    """
    n = G.number_of_nodes()
    if n == 0:
        return float("nan")
    # Extract largest connected component
    comps = G.connected_components()
    if not comps:
        return float("nan")
    largest = max(comps, key=len)
    if len(largest) < 2:
        return 0.0
    # Build adjacency matrix for the component
    node_list = list(largest)
    idx_map = {node: idx for idx, node in enumerate(node_list)}
    rows = []
    cols = []
    for u in node_list:
        for v in G._adj[u]:
            if v in idx_map:
                rows.append(idx_map[u])
                cols.append(idx_map[v])
    data = np.ones(len(rows), dtype=float)
    n_lcc = len(node_list)
    adj = csr_matrix((data, (rows, cols)), shape=(n_lcc, n_lcc))
    # Compute Laplacian eigenvalues
    L = laplacian(adj, normed=False)
    # Convert to dense for eigenvalues; for moderate graphs this is acceptable
    L_dense = L.toarray()
    eigvals = np.linalg.eigvalsh(L_dense)
    # Remove zero eigenvalues (tolerance)
    non_zero = [lam for lam in eigvals if lam > 1e-9]
    if not non_zero:
        return 0.0
    R_G = n_lcc * float(np.sum(1.0 / np.array(non_zero)))
    return float((n_lcc - 1) / R_G)


def compute_basic_metrics(G: SimpleGraph) -> Dict[str, Union[float, int]]:
    """Compute basic network metrics for a SimpleGraph."""
    n = G.number_of_nodes()
    m = G.number_of_edges()
    metrics = {
        "n_nodes": n,
        "n_edges": m,
        "avg_degree": float(2 * m / n) if n > 0 else float("nan"),
        "giant_fraction": float("nan"),
        "avg_path_length": float("nan"),
        "avg_clustering": float("nan"),
    }
    if n == 0:
        return metrics
    comps = G.connected_components()
    if not comps:
        metrics["giant_fraction"] = float("nan")
    else:
        largest = max(comps, key=len)
        metrics["giant_fraction"] = len(largest) / n
        # Compute average shortest path length on the largest component
        if len(largest) > 1:
            # Build adjacency for this component
            node_list = list(largest)
            idx_map = {node: idx for idx, node in enumerate(node_list)}
            rows = []
            cols = []
            for u in node_list:
                for v in G._adj[u]:
                    if v in idx_map:
                        rows.append(idx_map[u])
                        cols.append(idx_map[v])
            data = np.ones(len(rows), dtype=float)
            n_lcc = len(node_list)
            adj = csr_matrix((data, (rows, cols)), shape=(n_lcc, n_lcc))
            dist_matrix = shortest_path(adj, method='D', directed=False, unweighted=True)
            # Only finite distances
            finite_dists = dist_matrix[np.isfinite(dist_matrix)]
            if len(finite_dists) > 1:
                # Exclude zero-length distances by ignoring diagonal
                finite_dists = finite_dists[finite_dists > 0]
                metrics["avg_path_length"] = float(np.mean(finite_dists))
    # Clustering coefficient
    # For each node, count triangles and possible triples
    clustering_values: List[float] = []
    for u in G.nodes():
        neighbors = G._adj[u]
        k = len(neighbors)
        if k < 2:
            continue
        # Count pairs of neighbors that are connected
        tri = 0
        neighbors_list = list(neighbors)
        for i in range(k):
            for j in range(i + 1, k):
                v = neighbors_list[i]
                w = neighbors_list[j]
                if G.has_edge(v, w):
                    tri += 1
        clustering_values.append(tri / (k * (k - 1) / 2))
    if clustering_values:
        metrics["avg_clustering"] = float(np.mean(clustering_values))
    else:
        metrics["avg_clustering"] = 0.0
    return metrics


def simple_robustness_index(
    G: SimpleGraph,
    frac_remove: float = 0.2,
    strategy: str = "degree",
    seed: Optional[int] = None,
) -> float:
    """Compute a simple robustness proxy by removing a fraction of nodes.

    Parameters
    ----------
    G: SimpleGraph
        The graph to analyse.
    frac_remove: float
        Fraction of nodes to remove (0 < frac <= 1).
    strategy: str
        Removal strategy: "degree" removes the highest degree nodes;
        "random" removes uniformly at random.
    seed: int or None
        Random seed for reproducibility in the random case.

    Returns
    -------
    float
        Fraction of nodes in the largest connected component after
        removal relative to the original number of nodes.
    """
    n = G.number_of_nodes()
    if n == 0:
        return float("nan")
    n_remove = max(1, int(frac_remove * n))
    if n_remove >= n:
        n_remove = n - 1
    if strategy == "degree":
        degs = G.degree()
        # sort by degree descending; in tie, by node id to ensure determinism
        nodes_sorted = sorted(degs.items(), key=lambda x: (-x[1], x[0]))
        to_remove = [nid for nid, _ in nodes_sorted[:n_remove]]
    elif strategy == "random":
        rng = random.Random(seed)
        to_remove = rng.sample(G.nodes(), n_remove)
    else:
        raise ValueError("strategy must be 'degree' or 'random'")
    H = G.copy()
    H.remove_nodes_from(to_remove)
    if H.number_of_nodes() == 0:
        return 0.0
    comps = H.connected_components()
    if not comps:
        return 0.0
    largest = max(comps, key=len)
    return len(largest) / n


def compute_dataset_summary(
    graphs: Dict[str, SimpleGraph],
    frac_remove: float = 0.2,
    random_runs: int = 10,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Compute a summary DataFrame for a set of graphs."""
    rows = []
    for name, G in graphs.items():
        metrics = compute_basic_metrics(G)
        r_T = robustness_indicator_rT(G)
        C_G = effective_graph_conductance_CG(G)
        rob_degree = simple_robustness_index(G, frac_remove=frac_remove, strategy="degree")
        # Average random removals
        rob_random_total = 0.0
        for i in range(max(1, random_runs)):
            run_seed = None if seed is None else seed + i
            rob_random_total += simple_robustness_index(G, frac_remove=frac_remove, strategy="random", seed=run_seed)
        rob_random = rob_random_total / max(1, random_runs)
        row = {
            "name": name,
            **metrics,
            "r_T": r_T,
            "C_G": C_G,
            f"robustness_degree_{int(frac_remove*100)}pct": rob_degree,
            f"robustness_random_{int(frac_remove*100)}pct": rob_random,
        }
        rows.append(row)
    return pd.DataFrame(rows)
