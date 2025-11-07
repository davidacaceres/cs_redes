# build_majority_illusion_fig2_ipynb.py
# Crea el notebook "Majority-Illusion-Fig2.ipynb" con la implementación y documentación solicitadas.

import nbformat as nbf
from textwrap import dedent

nb = nbf.v4.new_notebook()
cells = []

# ---- Portada y caption ----
cells.append(nbf.v4.new_markdown_cell(dedent(r"""
# The "Majority Illusion" in Social Networks — Fig. 2 (Reproducible Notebook)

**Caption to replicate (from the paper):**  
*Plots show the magnitude of the illusion in scale-free networks as a function of degree–attribute correlation* \( \rho_{kx} \) *and for different values of degree assortativity* \( r_{kk} \). *Each network has 10,000 nodes and degree distribution of the form* \( p(k)\sim k^{-\alpha} \). *The fraction of active nodes in all cases is 5%.* *The lines represent calculations using the statistical model of Eq (5).*

---

## Qué hace este notebook

- Construye **redes scale-free** con \(N=10{,}000\) nodos usando una secuencia de grados \(p(k)\propto k^{-\alpha}\).
- Ajusta la **asortatividad por grado** \(r_{kk}\) con *edge rewiring* que preserva la secuencia de grados del grafo simple.
- Asigna un **atributo binario** con **prevalencia fija** \(p=0.05\) y **correlación** \( \rho_{kx} \) objetivo con el grado (búsqueda por bisección en un sesgo \(k^\beta\)).
- Calcula \(P_{>1/2}\) (nodos con **más del 50%** de vecinos activos) **en la GCC**, excluyendo nodos de grado 0.
- Reproduce los **tres paneles** (α=2.1, 2.4, 3.1) **en una fila**, con los conjuntos de \(r_{kk}\) del paper.

> *En la figura del paper, las líneas representan el modelo teórico (Ec. 5). Aquí unimos con líneas los puntos empíricos de la simulación; puedes superponer el modelo cuando implementes esa ecuación.*
""")))

# ---- Imports ----
cells.append(nbf.v4.new_code_cell(dedent(r"""
import numpy as np
import networkx as nx
import scipy.sparse as sp
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import Dict, Tuple, List
""")))

# ---- Config y diseño ----
cells.append(nbf.v4.new_markdown_cell(dedent(r"""
## Diseño experimental (editables)

- **Tamaño** \(N=10{,}000\).
- **Exponente** \(\alpha \in \{2.1, 2.4, 3.1\}\).
- **rkk por panel**:  
  - α=2.1: −0.35, −0.25, −0.15, −0.05  
  - α=2.4: −0.20, −0.10, 0.00, 0.10, 0.20  
  - α=3.1: −0.15, −0.05, 0.00, 0.30
- **ρkx**: 12 puntos en [0.00, 0.55].
- **Prevalencia**: \(p=0.05\) (5%).
""")))

cells.append(nbf.v4.new_code_cell(dedent(r"""
@dataclass
class Config:
    N: int = 10_000
    alphas: Tuple[float,...] = (2.1, 2.4, 3.1)
    r_by_alpha: Dict[float, Tuple[float,...]] = None
    rho_grid: Tuple[float,...] = tuple(np.round(np.linspace(0.0, 0.55, 12), 2))
    p_global: float = 0.05             # 5%
    kmin: int = 1
    kmax: int | None = None            # truncamiento suave si None
    # Rewiring r_kk
    tol_r: float = 0.005
    max_rewire_steps_per_edge: float = 4.0
    # Asignación rho_kx
    tol_rho: float = 0.015
    max_bias_bisection: int = 22
    # Medición
    sample_nodes: int | None = None    # e.g., 4000 para acelerar
    trials_per_rho: int = 1
    rng_seed: int = 123

CFG = Config()
CFG.r_by_alpha = {
    2.1: (-0.35, -0.25, -0.15, -0.05),
    2.4: (-0.20, -0.10,  0.00,  0.10,  0.20),
    3.1: (-0.15, -0.05,  0.00,  0.30),
}
CFG
""")))

# ---- Red scale-free ----
cells.append(nbf.v4.new_markdown_cell(dedent(r"""
## Generación de red scale-free

1. Muestreamos una **secuencia de grados** \(k_i\) de \(p(k)\propto k^{-\alpha}\) en \([k_{\min}, k_{\max}]\).  
2. Construimos un grafo **simple** con `configuration_model`, colapsando múltiples aristas y lazos.  
   Trabajamos con la secuencia **observada** del grafo simple (es la que se preserva en el rewiring).
""")))

cells.append(nbf.v4.new_code_cell(dedent(r"""
def sample_powerlaw_degrees(N:int, alpha:float, kmin:int=1, kmax:int|None=None, rng=None):
    if rng is None: rng = np.random.default_rng()
    if kmax is None:
        # evitar súper-hubs extremos: tope suave ~10·√N
        kmax = int(max(kmin, min(N-1, (N**0.5)*10)))
    kmax = int(min(kmax, N-1))
    ks = np.arange(kmin, kmax+1, dtype=int)
    pk = ks.astype(float)**(-alpha); pk /= pk.sum()
    deg = rng.choice(ks, size=N, replace=True, p=pk)
    if deg.sum() % 2 == 1:
        i = rng.integers(0, N); deg[i] = min(deg[i]+1, kmax)
    deg = np.maximum(deg, kmin)
    return deg

def config_graph_from_seq(deg: np.ndarray, seed: int = 42) -> nx.Graph:
    Gm = nx.configuration_model(deg, seed=seed)
    G = nx.Graph(Gm)              # simple
    G.remove_edges_from(nx.selfloop_edges(G))
    return G
""")))

# ---- Rewiring a r_kk ----
cells.append(nbf.v4.new_markdown_cell(dedent(r"""
## Rewiring para alcanzar \(r_{kk}\)

Sea \(A=\sum_{(i,j)\in E} k_i k_j\). Con grados fijos, \(B=\tfrac12\sum k_i^2\) y \(C=\tfrac12\sum k_i^3\) son constantes, y

\[
r_{kk}=\frac{A/M-(B/M)^2}{C/M-(B/M)^2},\qquad M=|E|.
\]

Un *edge swap* cambia \(A\) en \(\Delta A\). Elegimos swaps que **mueven \(A\)** en la dirección del objetivo y evitamos multi-aristas/lazos.  
Esto permite **actualizar \(r_{kk}\) en O(1)** por swap.
""")))

cells.append(nbf.v4.new_code_cell(dedent(r"""
def _assort_constants_from_degrees(G: nx.Graph, deg: Dict[int,int] = None):
    if deg is None: deg = dict(G.degree())
    M = G.number_of_edges()
    deg_arr = np.array(list(deg.values()), dtype=float)
    B = 0.5 * np.sum(deg_arr**2)
    C = 0.5 * np.sum(deg_arr**3)
    A = 0
    for u, v in G.edges():
        A += deg[u]*deg[v]
    denom = (C/M - (B/M)**2)
    return A, B, C, M, denom, deg

def assort_from_A(A: float, B: float, C: float, M: float) -> float:
    return (A/M - (B/M)**2) / (C/M - (B/M)**2)

def rewire_to_rkk(G: nx.Graph, r_target: float, tol: float = 0.01,
                  max_rewire_steps_per_edge: float = 3.0, seed: int = 42,
                  verbose: bool = False) -> float:
    rng = np.random.default_rng(seed)
    A, B, C, M, denom, deg = _assort_constants_from_degrees(G)
    if denom <= 0:
        return nx.degree_assortativity_coefficient(G)
    r = assort_from_A(A, B, C, M)
    edges: List[Tuple[int,int]] = list(G.edges())
    adj = {u: set(G.neighbors(u)) for u in G.nodes()}
    steps = int(max_rewire_steps_per_edge * M)
    direction = 1 if r_target > r else -1
    no_improve = 0
    for t in range(steps):
        i = rng.integers(len(edges)); j = rng.integers(len(edges))
        if i == j: 
            continue
        a, b = edges[i]; c, d = edges[j]
        if len({a,b,c,d}) < 4: 
            continue
        valid1 = (c not in adj[a]) and (d not in adj[b]) and (a != c) and (b != d)
        valid2 = (d not in adj[a]) and (c not in adj[b]) and (a != d) and (b != c)
        if not (valid1 or valid2):
            continue
        ka, kb, kc, kd = deg[a], deg[b], deg[c], deg[d]
        current = ka*kb + kc*kd
        delta1 = (ka*kc + kb*kd) - current
        delta2 = (ka*kd + kb*kc) - current
        cand = []
        if valid1: cand.append(("ac_bd", delta1))
        if valid2: cand.append(("ad_bc", delta2))
        if not cand:
            continue
        label, delta = max(cand, key=lambda x: x[1]) if direction>0 else min(cand, key=lambda x: x[1])
        # early rejection si no mejora en la dirección deseada
        if (direction > 0 and delta <= 0) or (direction < 0 and delta >= 0):
            no_improve += 1
            if no_improve > 5000:
                r = assort_from_A(A, B, C, M)
                direction = 1 if r_target > r else -1
                no_improve = 0
            continue
        # aplica el swap manteniendo simpleza
        G.remove_edge(a, b); adj[a].remove(b); adj[b].remove(a)
        G.remove_edge(c, d); adj[c].remove(d); adj[d].remove(c)
        if label == "ac_bd":
            G.add_edge(a, c); adj[a].add(c); adj[c].add(a)
            G.add_edge(b, d); adj[b].add(d); adj[d].add(b)
            edges[i] = (a, c); edges[j] = (b, d)
        else:
            G.add_edge(a, d); adj[a].add(d); adj[d].add(a)
            G.add_edge(b, c); adj[b].add(c); adj[c].add(b)
            edges[i] = (a, d); edges[j] = (b, c)
        A += delta
        r = assort_from_A(A, B, C, M)
        direction = 1 if r_target > r else -1
        if verbose and (t % (M//2 + 1) == 0):
            print(f"[{t}/{steps}] r≈{r:.4f} (target {r_target:+.2f})")
        if abs(r - r_target) <= tol:
            break
    return r
""")))

# ---- Asignación atributo (rho_kx, p fijo) ----
cells.append(nbf.v4.new_markdown_cell(dedent(r"""
## Asignación del atributo: \( \rho_{kx} \) objetivo con prevalencia fija \(p\)

Tomamos \(x\in\{0,1\}^N\) con \( \mathbb{E}[x]=p \).  
Usamos **bisección** sobre un sesgo \(k^\beta\) para muestrear exactamente \(pN\) activos con probabilidad \(\propto k^\beta\), hasta acercar \(\rho_{kx}\) al objetivo.
""")))

cells.append(nbf.v4.new_code_cell(dedent(r"""
def assign_attribute_with_rho_and_p(k: np.ndarray, rho_target: float, p: float,
                                    tol: float = 0.015, max_iter: int = 22,
                                    rng=None):
    if rng is None: rng = np.random.default_rng()
    N = len(k); m = int(round(p * N))
    lo, hi = -8.0, +8.0
    x_best, rho_best = None, -1.0
    for _ in range(max_iter):
        beta = 0.5*(lo + hi)
        w = k.astype(float)**beta
        w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
        w = w if w.sum() > 0 else np.ones_like(w)
        w = w / w.sum()
        idx = rng.choice(N, size=m, replace=False, p=w)
        x = np.zeros(N, dtype=np.int8); x[idx] = 1
        rho = np.corrcoef(k, x)[0,1]
        x_best, rho_best = x, rho
        if abs(rho - rho_target) <= tol: break
        if rho < rho_target: lo = beta
        else: hi = beta
    return x_best, rho_best
""")))

# ---- P>1/2 en la GCC ----
cells.append(nbf.v4.new_markdown_cell(dedent(r"""
## Medición de \(P_{>1/2}\) en la GCC

Calculamos \(A\,x\) con CSR, contamos vecinos activos y evaluamos si superan la mitad del grado.
Se excluyen nodos con \(k=0\) y se considera solo la **GCC**.  
Se puede muestrear una cantidad fija de nodos para acelerar (parámetro `sample_nodes`).
""")))

cells.append(nbf.v4.new_code_cell(dedent(r"""
def prob_majority_in_gcc(G: nx.Graph, x: np.ndarray, sample_nodes: int | None = None) -> float:
    GCC = max(nx.connected_components(G), key=len)
    H = G.subgraph(GCC).copy()
    nodes = np.array(list(H.nodes()), dtype=int)
    xH = x[nodes]
    A = nx.to_scipy_sparse_array(H, format="csr", dtype=np.int8)
    deg = np.asarray(A.sum(1)).ravel()
    ok = deg > 0
    if sample_nodes is not None and sample_nodes < ok.sum():
        rng = np.random.default_rng(123)
        idx = np.where(ok)[0]
        choose = rng.choice(idx, size=sample_nodes, replace=False)
        mask = np.zeros_like(ok, dtype=bool); mask[choose] = True
        ok = mask
    counts = (A @ xH)
    return (counts[ok] > 0.5*deg[ok]).mean()
""")))

# ---- Pipeline por panel ----
cells.append(nbf.v4.new_markdown_cell(dedent(r"""
## Pipeline por panel (α fijo)

Para cada \(r_{kk}\) objetivo: se genera una red, se ajusta \(r_{kk}\) por rewiring y se reutiliza esa red para todos los \(\rho_{kx}\).
""")))

cells.append(nbf.v4.new_code_cell(dedent(r"""
def run_panel(alpha: float, r_targets: Tuple[float,...], cfg: Config = CFG):
    rng = np.random.default_rng(cfg.rng_seed)
    results = {}
    r_measured = {}
    for r_target in r_targets:
        deg = sample_powerlaw_degrees(cfg.N, alpha, cfg.kmin, cfg.kmax, rng)
        G = config_graph_from_seq(deg, seed=cfg.rng_seed)
        r_real = rewire_to_rkk(G, r_target, tol=cfg.tol_r,
                               max_rewire_steps_per_edge=cfg.max_rewire_steps_per_edge,
                               seed=cfg.rng_seed, verbose=False)
        r_measured[r_target] = r_real
        k = np.array([d for _, d in sorted(G.degree(), key=lambda x: x[0])], dtype=int)
        vals = []
        for rho_target in cfg.rho_grid:
            acc = []
            for _ in range(cfg.trials_per_rho):
                x, rho_real = assign_attribute_with_rho_and_p(k, rho_target, cfg.p_global,
                                                              tol=cfg.tol_rho,
                                                              max_iter=cfg.max_bias_bisection,
                                                              rng=rng)
                p = prob_majority_in_gcc(G, x, sample_nodes=cfg.sample_nodes)
                acc.append(p)
            vals.append(float(np.mean(acc)))
        results[r_target] = vals
    return results, r_measured
""")))

# ---- Tres paneles en una fila ----
cells.append(nbf.v4.new_markdown_cell(dedent(r"""
## Gráfico final: tres paneles en una fila
""")))

cells.append(nbf.v4.new_code_cell(dedent(r"""
def run_all_panels(cfg: Config = CFG):
    results_by_alpha = {}
    r_meas_by_alpha = {}
    for a in cfg.alphas:
        res, r_meas = run_panel(a, cfg.r_by_alpha[a], cfg)
        results_by_alpha[a] = res
        r_meas_by_alpha[a] = r_meas
    return results_by_alpha, r_meas_by_alpha

def plot_three_panels(results_by_alpha, r_meas_by_alpha, cfg: Config = CFG):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4), dpi=120, sharex=True, sharey=True)
    marker_sets = {
        2.1: ["o", "^", "s", "x"],      # 4 curvas
        2.4: ["o", "^", "s", "D", "x"], # 5 curvas
        3.1: ["o", "^", "s", "x"],      # 4 curvas
    }
    for ax, a in zip(axes, cfg.alphas):
        r_targets = cfg.r_by_alpha[a]
        markers = marker_sets[a]
        for i, r_target in enumerate(r_targets):
            lab = rf"$r_{{kk}}={r_meas_by_alpha[a][r_target]:+.2f}$"
            ax.plot(cfg.rho_grid, results_by_alpha[a][r_target],
                    marker=markers[i % len(markers)], linewidth=1.8, markersize=5.5, label=lab)
        ax.set_title(rf"(a) $\alpha=2.1$" if a==2.1 else (rf"(b) $\alpha=2.4$" if a==2.4 else rf"(c) $\alpha=3.1$"))
        ax.grid(True, alpha=0.25, linestyle="--")
        if a==2.1:
            ax.set_ylabel(r"Probability of majority, $P_{>1/2}$")
        ax.set_xlabel(r"$k\text{–}x$ correlation, $\rho_{kx}$")
        ax.set_xlim(min(cfg.rho_grid)-0.02, max(cfg.rho_grid)+0.02)
        ax.set_ylim(0, 1.0)
        ax.legend(frameon=True, loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.show()
""")))

# ---- Ejecutar ----
cells.append(nbf.v4.new_markdown_cell(dedent(r"""
## Ejecutar todo

**Costo computacional:** con \(N=10{,}000\) puede demorar.  
Para acelerar, antes de ejecutar puedes fijar:
```python
CFG.sample_nodes = 4000          # muestrear nodos al medir P_{>1/2}
CFG.max_rewire_steps_per_edge = 2.0
""")))

cells.append(nbf.v4.new_code_cell(dedent(r"""

Aceleración opcional (descomenta si lo deseas)
CFG.sample_nodes = 4000
CFG.max_rewire_steps_per_edge = 2.0

results_by_alpha, r_meas_by_alpha = run_all_panels(CFG)
plot_three_panels(results_by_alpha, r_meas_by_alpha, CFG)
""")))

nb['cells'] = cells
nb['metadata'] = {
"kernelspec": {"name": "python3", "language": "python", "display_name": "Python 3"},
"language_info": {"name": "python", "pygments_lexer": "ipython3"}
}
nb['nbformat'] = 4
nb['nbformat_minor'] = 5

out = "Majority-Illusion-scale-free-Fig2.ipynb"
with open(out, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Notebook generado: {out}")


# ---- Escribir el archivo .ipynb ----

with open("Majority-Illusion-Fig2.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("OK: Majority-Illusion-Fig2.ipynb creado en el directorio actual.")


