"""
tarea3_module
-------------------

This module provides utilities for downloading, parsing and analysing
public transport networks without relying on external graph libraries.

The central class in this module is :class:`SimpleGraph`, a minimal
graph structure implemented using adjacency sets.  It supports basic
operations such as adding nodes and edges, copying, removing nodes,
and computing degrees.  A suite of helper functions builds
``SimpleGraph`` instances from common file formats (CSV, JSON,
GraphML/GEXF converted to adjacency lists), computes robustness metrics
including the cyclomatic index ``r_T``, effective graph conductance
``C_G``, as well as basic structural metrics and a simple robustness
proxy.

These functions avoid dependencies on the ``networkx`` library,
which may not be available in some execution environments.  They
instead use ``numpy`` and ``scipy.sparse.csgraph`` where appropriate
to perform graph analysis.
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

try:
    import gdown  # type: ignore
    _HAS_GDOWN = True
except Exception:
    _HAS_GDOWN = False

import requests


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


def download_file(url: str, dest: Path, chunk_size: int = 2 ** 20) -> Path:
    """Download a file via HTTP and write it to disk.

    If the file already exists at the destination it is not downloaded again.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(dest, "wb") as fh:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                fh.write(chunk)
    return dest


def download_from_drive(file_id: str, dest: Path) -> Path:
    """Download a file from Google Drive using gdown or requests.

    Parameters
    ----------
    file_id : str
        The ID part of the Google Drive sharing link.
    dest : Path
        Destination path on disk.

    Returns
    -------
    Path
        The path where the file was saved.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    if _HAS_GDOWN:
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, str(dest), quiet=False)
        return dest
    # Fallback: direct download may require confirmation token which we do not handle
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    return download_file(url, dest)


def extract_zip(zip_path: Path, dest_dir: Path) -> None:
    """Extract a ZIP archive into a directory."""
    import zipfile
    dest_dir.mkdir(parents=True, exist_ok=True)
    if any(dest_dir.iterdir()):
        return
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)


def build_graph_from_kujala_city(city_dir: Path) -> Tuple[SimpleGraph, pd.DataFrame]:
    """Construct a SimpleGraph from a Kujala city directory.

    Each city directory must contain ``network_nodes.csv`` and
    ``network_combined.csv`` separated with ``;``.  This function
    reads the node file, maps columns according to a set of possible
    names for the identifier, latitude, longitude and name, and
    constructs an undirected graph of stops.  Node attributes are
    preserved in a returned DataFrame.
    """
    nodes_csv = city_dir / "network_nodes.csv"
    edges_csv = city_dir / "network_combined.csv"

    print("Path nodo ",nodes_csv)
    print("Path edges ", edges_csv)
    if not nodes_csv.exists() or not edges_csv.exists():
        print("Archivo no encontrado ", nodes_csv)
        raise FileNotFoundError(f"Missing required files in {city_dir}")
    nodes_df = pd.read_csv(nodes_csv, sep=";")
    # Determine identifier and attribute columns
    id_col = None
    for cand in ["stop_I", "node_I", "node_id", "stop_id", "stop_i"]:
        if cand in nodes_df.columns:
            id_col = cand
            break
    if id_col is None:
        raise ValueError(f"No identifier column found in {nodes_csv}")
    lat_col = next((c for c in nodes_df.columns if c.lower() in {"lat", "latitude"}), None)
    lon_col = next((c for c in nodes_df.columns if c.lower() in {"lon", "longitude"}), None)
    name_col = next((c for c in nodes_df.columns if "name" in c.lower()), None)
    G = SimpleGraph(name=f"kujala_{city_dir.name}")
    node_attrs: List[Dict[str, Any]] = []
    for _, row in nodes_df.iterrows():
        nid = row[id_col]
        attrs: Dict[str, Any] = {}
        if lat_col is not None and pd.notna(row[lat_col]):
            attrs["lat"] = float(row[lat_col])
        if lon_col is not None and pd.notna(row[lon_col]):
            attrs["lon"] = float(row[lon_col])
        if name_col is not None and pd.notna(row[name_col]):
            attrs["name"] = str(row[name_col])
        G.add_node(nid, **attrs)
        node_attrs.append({"node_id": nid, **attrs})
    edges_df = pd.read_csv(edges_csv, sep=";")
    from_col = None
    to_col = None
    for cand in ["from_stop_I", "from_node_I", "from_id", "from_stop", "u"]:
        if cand in edges_df.columns:
            from_col = cand
            break
    for cand in ["to_stop_I", "to_node_I", "to_id", "to_stop", "v"]:
        if cand in edges_df.columns:
            to_col = cand
            break
    if from_col is None or to_col is None:
        raise ValueError(f"Could not find edge endpoints in {edges_csv}")
    for _, row in edges_df.iterrows():
        u = row[from_col]
        v = row[to_col]
        if pd.isna(u) or pd.isna(v):
            continue
        # Convert to int if possible
        try:
            u_int = int(u)
            v_int = int(v)
        except Exception:
            u_int = u
            v_int = v
        G.add_edge(u_int, v_int)
    nodes_df_out = pd.DataFrame(node_attrs)
    return G, nodes_df_out


def load_kujala_dataset(root: Path) -> Tuple[Dict[str, SimpleGraph], Dict[str, pd.DataFrame]]:
    """Load all city networks from the Kujala dataset directory."""
    graphs: Dict[str, SimpleGraph] = {}
    nodes: Dict[str, pd.DataFrame] = {}
    if not root.exists():
        return graphs, nodes
    for city_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        print('Buscando ciudad ',city_dir)
        try:
            G, df_nodes = build_graph_from_kujala_city(city_dir)
            graphs[city_dir.name] = G
            nodes[city_dir.name] = df_nodes
        except Exception:
            continue
    return graphs, nodes


def load_metro51_from_folder(root: Path) -> Tuple[Dict[str, SimpleGraph], Dict[str, pd.DataFrame]]:
    """Load all networks from the 51 metro dataset folder.

    This function iterates over all files under ``root`` and attempts to
    parse those that represent graphs.  JSON files must have ``nodes``
    and ``edges`` lists with interpretable field names; GraphML and
    GEXF files are ignored because we cannot parse them without
    external libraries in this environment.  GPickle files are also
    skipped.  Only JSON is supported by default.
    """
    graphs: Dict[str, SimpleGraph] = {}
    nodes: Dict[str, pd.DataFrame] = {}
    if not root.exists():
        return graphs, nodes
    files = [p for p in root.rglob("*") if p.is_file()]
    for path in sorted(files):
        name = path.stem
        ext = path.suffix.lower()
        try:
            if ext == ".json":
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if not isinstance(data, dict):
                    continue
                nodes_list = data.get("nodes") or data.get("Nodes")
                edges_list = data.get("edges") or data.get("Edges")
                if not nodes_list or not edges_list:
                    continue
                # Determine node id key
                node_id_key = None
                sample_node = nodes_list[0]
                for cand in ["id", "node_id", "node", "stop_id", "stop", "idNode", "key"]:
                    if cand in sample_node:
                        node_id_key = cand
                        break
                if node_id_key is None:
                    continue
                G = SimpleGraph(name=name)
                node_attrs_rows: List[Dict[str, Any]] = []
                for nd in nodes_list:
                    nid = nd[node_id_key]
                    attrs = {k: v for k, v in nd.items() if k != node_id_key}
                    G.add_node(nid, **attrs)
                    row = {"node_id": nid}
                    row.update(attrs)
                    node_attrs_rows.append(row)
                # Determine edge keys
                sample_edge = edges_list[0]
                u_key = None
                v_key = None
                for cand in ["source", "from", "u", "from_id", "i"]:
                    if cand in sample_edge:
                        u_key = cand
                        break
                for cand in ["target", "to", "v", "to_id", "j"]:
                    if cand in sample_edge:
                        v_key = cand
                        break
                if u_key is None or v_key is None:
                    continue
                for e in edges_list:
                    u = e[u_key]
                    v = e[v_key]
                    G.add_edge(u, v)
                graphs[name] = G
                nodes[name] = pd.DataFrame(node_attrs_rows)
        except Exception:
            # Skip files that cannot be parsed
            continue
    return graphs, nodes


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
