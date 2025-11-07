#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fig. 2 — Ilusión de Mayoría en redes scale-free (script standalone para PyCharm)
-------------------------------------------------------------------------------
Reproduce únicamente el experimento de la Figura 2 del paper:
  "The Majority Illusion in Social Networks" (Lerman, Yan, Wu, 2016)

Estructura del experimento:
- 3 paneles por exponente de la ley de potencia (alpha ∈ {2.1, 2.4, 3.1})
- En cada panel, varias curvas por asortatividad por grado r_kk
- Eje X: correlación grado–atributo (rho_kx)
- Eje Y: probabilidad de mayoría P_{>1/2} (nodos con >50% de vecinos activos)

Optimizado para tiempo de cómputo:
- Construcción de redes por modelo de configuración (GCC grande)
- Edge-rewiring para r_kk (preserva secuencia de grados)
- Asignación rápida de rho_kx (búsqueda binaria en sesgo por grado; NO hace swapping literal)
- Cálculo de P_{>1/2} vectorizado con matriz dispersa (CSR)

Requisitos:
    pip install numpy scipy pandas networkx matplotlib tqdm

Uso:
    python fig2_majority_illusion.py
    python fig2_majority_illusion.py --N 5000 --rho_grid 0,0.2,0.4,0.6 --warm_start 1
    python fig2_majority_illusion.py --out_png Fig2.png --out_csv Fig2_results.csv

Novedades de esta versión:
- Rewiring con límite dinámico: steps = rewire_factor × m (m = #aristas), o fijo si pasas --max_rewire_steps>0.
- Logging opcional (verbose) de: modo, límite usado, pasos consumidos y swaps aceptados.
- Salidas con carpeta/tag para no sobreescribir.
"""

from __future__ import annotations
import argparse
import random
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.sparse import csr_matrix

# tqdm es opcional; si no está, definimos un reemplazo "inofensivo"
try:
    from tqdm import tqdm
    def _log(msg: str):
        try:
            tqdm.write(msg)
        except Exception:
            print(msg)
except Exception:
    tqdm = lambda x, **k: x  # fallback si no está tqdm
    def _log(msg: str):
        print(msg)

# ----------------------------------------------------------------------------- #
# Utilidades generales
# ----------------------------------------------------------------------------- #

def set_seeds(seed: int = 123) -> None:
    """Fija las semillas de Python y NumPy para reproducibilidad."""
    random.seed(seed)
    np.random.seed(seed)

def average_degree(G: nx.Graph) -> float:
    """Retorna el grado promedio del grafo."""
    return sum(dict(G.degree()).values()) / G.number_of_nodes()

# ----------------------------------------------------------------------------- #
# 1) Generación de secuencia de grados y grafo de configuración
# ----------------------------------------------------------------------------- #

def sample_powerlaw_degrees(
    N: int,
    alpha: float = 2.4,
    k_min: int = 2,
    k_max: int | None = None
) -> List[int]:
    """
    Muestra N grados con distribución ~ k^(-alpha), usando inverse transform (continuo).
    - k_min >= 2 ayuda a que la red tenga menos hojas y mayor conectividad.
    - k_max por defecto ~ sqrt(N) para evitar hubs demasiado dominantes (numéricamente estable).
    - Asegura suma par (necesario para el modelo de configuración).
    """
    if k_max is None:
        k_max = max(k_min + 1, int(np.sqrt(N)))
    a = alpha
    r = np.random.random(size=N)
    low = k_min ** (1 - a)
    high = k_max ** (1 - a)
    ks = (r * (high - low) + low) ** (1 / (1 - a))
    ks = np.clip(np.rint(ks), k_min, k_max).astype(int)
    # suma par para configuration_model
    if int(ks.sum()) % 2 == 1:
        ks[np.random.randint(0, N)] += 1
    return ks.tolist()

def configuration_graph_from_degrees_connected(
    deg_seq: Sequence[int],
    min_gcc_frac: float = 0.95,
    max_tries: int = 30
) -> nx.Graph:
    """
    Construye un grafo simple a partir de la secuencia de grados (modelo de configuración),
    elimina lazos y multienlaces, y reintenta hasta que la componente conexa gigante (GCC)
    tenga al menos min_gcc_frac del total de nodos.
    """
    last_G = None
    for _ in range(max_tries):
        M = nx.configuration_model(deg_seq, seed=np.random.randint(1, 1_000_000))
        G = nx.Graph(M)                           # convierte a simple graph
        G.remove_edges_from(nx.selfloop_edges(G)) # quita self-loops
        last_G = G
        if nx.is_connected(G):
            return G
        gcc = max(nx.connected_components(G), key=len)
        if len(gcc) / len(deg_seq) >= min_gcc_frac:
            return G.subgraph(gcc).copy()
    # último recurso: devolver la GCC del último intento
    if last_G is None:
        raise RuntimeError("No se pudo construir el grafo de configuración.")
    gcc = max(nx.connected_components(last_G), key=len)
    return last_G.subgraph(gcc).copy()

# ----------------------------------------------------------------------------- #
# 2) Rewiring para imponer asortatividad r_kk (preservando grados)
# ----------------------------------------------------------------------------- #

def assortativity_targeted_rewire(
    G: nx.Graph,
    target_r: float = -0.2,
    max_steps: int = 30000,
    tol: float = 0.02
) -> Tuple[nx.Graph, float, int, int]:
    """
    Empuja r_kk hacia target_r con swaps que preservan grados.
    Se detiene por tolerancia (|r - target_r| <= tol); 'max_steps'
    es solo un tope de seguridad.
    Retorna además:
      - steps_used: iteraciones realizadas del bucle
      - swaps_accepted: cantidad de swaps que se aplicaron
    """
    G = G.copy()
    try:
        r = nx.degree_assortativity_coefficient(G)
    except Exception:
        r = 0.0
    edges = list(G.edges())
    m = len(edges)
    if m < 2:
        return G, r, 0, 0

    steps_used = 0
    swaps_accepted = 0

    for _ in range(max_steps):
        steps_used += 1
        if abs(r - target_r) <= tol:
            break
        (u, v) = edges[np.random.randint(m)]
        (x, y) = edges[np.random.randint(m)]
        if len({u, v, x, y}) < 4:
            continue

        # evaluar ambas formas de reempatar
        candidates = [((u, x), (v, y)), ((u, y), (v, x))]
        best = None
        best_err = abs(r - target_r)

        for (a, b), (c, d) in candidates:
            if a == b or c == d:        # evita lazos
                continue
            if G.has_edge(a, b) or G.has_edge(c, d):  # evita multienlaces
                continue

            # aplicar-calc-revert para evaluar
            G.remove_edge(u, v); G.remove_edge(x, y)
            G.add_edge(a, b);    G.add_edge(c, d)
            try:
                new_r = nx.degree_assortativity_coefficient(G)
            except Exception:
                new_r = r
            err = abs(new_r - target_r)
            # revertir antes de probar el otro candidato
            G.remove_edge(a, b); G.remove_edge(c, d)
            G.add_edge(u, v);    G.add_edge(x, y)

            if err < best_err:
                best = ((a, b), (c, d), new_r)
                best_err = err

        # aceptar el mejor si mejora
        if best is not None:
            (a, b), (c, d), r = best
            G.remove_edge(u, v); G.remove_edge(x, y)
            G.add_edge(a, b);    G.add_edge(c, d)
            swaps_accepted += 1
            edges = list(G.edges())
            m = len(edges)

    return G, r, steps_used, swaps_accepted

# ----------------------------------------------------------------------------- #
# 3) Asignación rápida de rho_kx (aprox. al "attribute swapping")
# ----------------------------------------------------------------------------- #

def fast_assign_rho(
    G: nx.Graph,
    p_active: float = 0.05,
    rho_target: float = 0.3,
    tol: float = 0.01,
    max_iter: int = 25
):
    """
    Aproxima una asignación de atributos x∈{0,1} con fracción activa p_active y
    correlación grado–atributo rho_kx ≈ rho_target. Método:
      - Define un "sesgo por grado" con exponente β (≥0).
      - Búsqueda binaria en β para subir/bajar la concentración de x=1 en grados altos/bajos.
      - Selecciona m nodos con mayor score como activos (m = round(p_active*N)).
    """
    nodes = list(G.nodes())
    degs = np.array([G.degree(n) for n in nodes], dtype=float)
    n = len(nodes)
    m = int(round(p_active * n))
    if m <= 0:
        x = np.zeros(n, dtype=int)
        return dict(zip(nodes, x.tolist())), 0.0

    positive = rho_target >= 0
    beta_lo, beta_hi = 0.0, 10.0
    best = None

    for _ in range(max_iter):
        beta = 0.5 * (beta_lo + beta_hi)
        # si rho_target>0 concentramos en grados altos; si <0, en grados bajos
        scores = np.power(degs + 1e-9, beta if positive else -beta)
        idx = np.argsort(scores)[::-1]
        chosen = idx[:m]
        x = np.zeros(n, dtype=int); x[chosen] = 1

        r = pearsonr(degs, x)[0] if (degs.std() > 0 and x.std() > 0) else 0.0
        cand = (abs(r - rho_target), r, x)
        if best is None or cand[0] < best[0]:
            best = cand

        if abs(r - rho_target) <= tol:
            break

        # ajusta el rango de búsqueda de β
        if positive:
            beta_lo = beta if r < rho_target else beta_lo
            beta_hi = beta if r >= rho_target else beta_hi
        else:
            beta_lo = beta if r > rho_target else beta_lo
            beta_hi = beta if r <= rho_target else beta_hi

    _, r_final, x_final = best
    return dict(zip(nodes, x_final.tolist())), float(r_final)

# ----------------------------------------------------------------------------- #
# 4) Probabilidad de mayoría P_{>1/2} (vectorizado con CSR)
# ----------------------------------------------------------------------------- #

def precompute_csr(G: nx.Graph, nodelist: List | None = None):
    """
    Devuelve:
      - nodelist: orden de nodos usado para construir la matriz
      - A: matriz de adyacencia CSR (rápida para multiplicación dispersa)
      - degs: vector de grados en el mismo orden
    """
    if nodelist is None:
        nodelist = list(G.nodes())
    A = nx.to_scipy_sparse_array(G, nodelist=nodelist, format='csr', dtype=np.int8)
    degs = np.asarray(A.sum(axis=1)).ravel()
    return nodelist, A, degs

def majority_illusion_fraction_csr(
    A: csr_matrix,
    degs: np.ndarray,
    x_vec: np.ndarray,
    phi: float = 0.5
) -> float:
    """
    Calcula P_{>phi} (por defecto phi=0.5): proporción de nodos cuyo
    número de vecinos activos supera phi * grado.
    Implementación: count( A @ x > phi*k ) / count(k>0)
    """
    neigh_act = A.dot(x_vec)
    eligible = degs > 0
    return float(np.mean(neigh_act[eligible] > phi * degs[eligible]))

# ----------------------------------------------------------------------------- #
# 5) Runner del experimento Fig. 2
# ----------------------------------------------------------------------------- #

def run_sf_fig2(
    N: int = 5000,
    alphas: Sequence[float] = (2.1, 2.4, 3.1),
    r_by_alpha: dict | None = None,
    rho_grid: Sequence[float] = (0.0, 0.2, 0.4, 0.6),
    p_active: float = 0.05,
    k_min: int = 2,
    max_rewire_steps: int = -1,         # -1: dinámico (factor × m)
    rewire_factor: float = 20.0,        # factor para límite dinámico
    tol_r: float = 0.02,
    tol_rho: float = 0.01,
    warm_start: bool = True,
    seed: int = 123,
    verbose: int = 1,
):
    """
    Orquesta el experimento:
      para cada alpha:
        - genera grafo de configuración (G0)
        - para cada r_target (r_kk objetivo):
            * rewire hasta ~r_target (límite fijo o dinámico)
            * precomputa CSR
            * para cada rho_target:
                · asigna x con fast_assign_rho (~rho_target, p_active fijo)
                · calcula P_{>1/2}
                · guarda fila de resultados (incluye métricas del rewiring)
    """
    set_seeds(seed)

    if r_by_alpha is None:
        r_by_alpha = {
            2.1: (-0.35, -0.20, -0.15, -0.05),
            2.4: (-0.20, -0.10,  0.00,  0.20),
            3.1: (-0.15, -0.05,  0.00,  0.30),
        }

    rows = []
    for alpha in tqdm(alphas, desc="α"):
        # Red base por panel
        degs = sample_powerlaw_degrees(N, alpha=alpha, k_min=k_min, k_max=int(np.sqrt(N)))
        G0 = configuration_graph_from_degrees_connected(degs, min_gcc_frac=0.95, max_tries=30)

        # Lista de objetivos de r_kk para este alpha
        r_list = list(r_by_alpha[alpha])
        # "warm_start": usa el grafo obtenido para el r anterior como base del siguiente
        if warm_start:
            r_list = sorted(r_list)

        G_cur = G0.copy()
        for r_target in tqdm(r_list, desc=f"r_kk (α={alpha})", leave=False):
            # Rewire para acercar r_kk
            r_start = nx.degree_assortativity_coefficient(G_cur)
            m_edges = G_cur.number_of_edges()
            steps_limit = int(max_rewire_steps if max_rewire_steps > 0 else rewire_factor * m_edges)
            mode = "fixed" if max_rewire_steps > 0 else f"dynamic({rewire_factor}×m={steps_limit})"

            G_cur, r_val, steps_used, swaps_acc = assortativity_targeted_rewire(
                G_cur, target_r=r_target, max_steps=steps_limit, tol=tol_r
            )

            if verbose:
                _log(f"[α={alpha}] r_target={r_target:+.2f} | modo={mode} | tol_r={tol_r:.3f} | "
                     f"r: {r_start:+.3f} → {r_val:+.3f} (Δ={abs(r_val-r_target):.3g}) | "
                     f"steps_used={steps_used}/{steps_limit} | swaps={swaps_acc}")

            # Precomputo CSR para acelerar cálculos repetidos
            nodelist, A, degs_arr = precompute_csr(G_cur)

            for rho_t in tqdm(rho_grid, desc="ρ_kx", leave=False):
                # Asignación rápida de atributos (fija p_active, apunta a rho_t)
                x_adj, rho_val = fast_assign_rho(
                    G_cur, p_active=p_active, rho_target=rho_t, tol=tol_rho
                )
                # vector x en el orden de 'nodelist'
                x_vec = np.array([x_adj[n] for n in nodelist], dtype=int)

                # Probabilidad de mayoría
                f = majority_illusion_fraction_csr(A, degs_arr, x_vec, phi=0.5)

                rows.append({
                    "alpha": alpha,
                    "r_target": r_target,
                    "assortativity": r_val,
                    "rho_target": rho_t,
                    "rho_kx": rho_val,
                    "illusion_frac": f,
                    "N_eff": G_cur.number_of_nodes(),
                    "k_avg": average_degree(G_cur),
                    # métricas de rewiring (para auditoría):
                    "rewire_mode": mode,
                    "m_edges": m_edges,
                    "steps_limit": steps_limit,
                    "steps_used": steps_used,
                    "swaps_accepted": swaps_acc,
                    "abs_error_r": abs(r_val - r_target),
                })

    return pd.DataFrame(rows)

# ----------------------------------------------------------------------------- #
# 6) Plot de la Fig. 2 (1x3 paneles)
# ----------------------------------------------------------------------------- #

def plot_sf_fig2(df: pd.DataFrame, r_by_alpha: dict, out_png: str | None = None) -> None:
    """
    Dibuja 3 subgráficos (uno por alpha).
    En cada panel: una curva por valor de r_kk, con marcadores sin relleno.
    Ejes y ticks fijados según la figura típica (X: 0..0.6, Y: 0..1).
    """
    marker_cycle = ['o', '^', 's', 'D', 'x', 'v', 'h']
    alphas_sorted = sorted(r_by_alpha.keys())
    fig, axes = plt.subplots(1, len(alphas_sorted), figsize=(6 * len(alphas_sorted), 4), sharey=True)
    if len(alphas_sorted) == 1:
        axes = [axes]

    for i, alpha in enumerate(alphas_sorted):
        ax = axes[i]
        sub = df[df['alpha'] == alpha]

        for j, r_t in enumerate(r_by_alpha[alpha]):
            part = sub[np.isclose(sub['r_target'], r_t)].sort_values('rho_kx')
            mk = marker_cycle[j % len(marker_cycle)]
            # Línea y marcadores “solo contorno”
            ax.plot(
                part['rho_kx'], part['illusion_frac'],
                marker=mk,
                markerfacecolor='none',  # sin relleno
                markeredgewidth=1.3,
                linestyle='-'
            )
            # entrada dummy para que aparezca bien en la leyenda
            ax.plot([], [], marker=mk, markerfacecolor='none', markeredgewidth=1.3,
                    linestyle='-', label=f"rₖₖ={r_t:+.2f}")

        ax.set_title(f"(α = {alpha})")
        ax.set_xlabel("k–x correlation,  $\\rho_{kx}$")
        if i == 0:
            ax.set_ylabel("Probability of majority,  $P_{>1/2}$")
        ax.grid(True, alpha=0.25)

        # Ticks y límites (como pediste)
        ax.set_xticks([0.0, 0.2, 0.4, 0.6]); ax.set_xlim(0.0, 0.6)
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]); ax.set_ylim(0.0, 1.0)
        ax.legend(frameon=False, loc="upper left")

    plt.tight_layout()
    if out_png:
        fig.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.show()

# ----------------------------------------------------------------------------- #
# 7) CLI / main
# ----------------------------------------------------------------------------- #

def parse_r_sets(arg: str):
    """Convierte un CSV de floats a tupla (por ejemplo: '-0.35,-0.2,0.0,0.2')."""
    if not arg:
        return tuple()
    return tuple(float(x.strip()) for x in arg.split(','))

def main():
    parser = argparse.ArgumentParser(description="Figura 2 — Ilusión de mayoría en redes scale-free (script PyCharm)")

    # Parámetros principales del experimento
    parser.add_argument('--N', type=int, default=5000, help='Tamaño de la red por panel (default: 5000)')
    parser.add_argument('--p_active', type=float, default=0.05, help='Fracción activa P(x=1) (default: 0.05)')
    parser.add_argument('--alphas', type=str, default='2.1,2.4,3.1', help='Exponentes power-law, CSV (default: 2.1,2.4,3.1)')
    parser.add_argument('--rho_grid', type=str, default='0.0,0.2,0.4,0.6', help='Valores de ρ_kx, CSV (default: 0.0,0.2,0.4,0.6)')

    # Conjuntos de r_kk por panel (puedes replicar exactamente los del paper)
    parser.add_argument('--r_2_1', type=str, default='-0.35,-0.20,-0.15,-0.05', help='r_kk para α=2.1 (CSV)')
    parser.add_argument('--r_2_4', type=str, default='-0.20,-0.10,0.00,0.20', help='r_kk para α=2.4 (CSV)')
    parser.add_argument('--r_3_1', type=str, default='-0.15,-0.05,0.00,0.30', help='r_kk para α=3.1 (CSV)')

    # Calidad/tiempo
    parser.add_argument('--k_min', type=int, default=2, help='Grado mínimo en la secuencia power-law (default: 2)')
    parser.add_argument('--tol_r', type=float, default=0.02, help='Tolerancia para r (default: 0.02)')
    parser.add_argument('--tol_rho', type=float, default=0.01, help='Tolerancia para ρ_kx (default: 0.01)')
    parser.add_argument('--warm_start', type=int, default=1, choices=[0,1], help='Warm-start del rewiring (0/1, default: 1)')
    parser.add_argument('--seed', type=int, default=123, help='Semilla RNG (default: 123)')

    # Salidas
    parser.add_argument('--out_png', type=str, default='Fig2_scale_free_majority_illusion.png', help='Ruta de salida PNG')
    parser.add_argument('--out_csv', type=str, default='Fig2_results.csv', help='Ruta de salida CSV')
    parser.add_argument('--out_dir', type=str, default='runs', help='Carpeta de salida (por corrida). Default: runs')
    parser.add_argument('--tag', type=str, default='', help='Etiqueta para nombres (si vacío, se usa timestamp)')

    # Rewire dinámico/fijo + verbose
    parser.add_argument('--max_rewire_steps', type=int, default=-1,help='Tope de pasos de rewiring. Si es -1, se usa rewire_factor * m (dinámico).')
    parser.add_argument('--rewire_factor', type=float, default=20.0,help='Factor para calcular pasos máx.: steps = factor * m (m = #aristas). Default: 20.0')
    parser.add_argument('--verbose', type=int, default=1, choices=[0,1], help='Imprime parámetros efectivos por cada r_kk (default: 1)')
    args = parser.parse_args()

    # Armado de carpeta y nombres únicos (no sobreescribir)
    from pathlib import Path
    import datetime
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = args.tag or ts
    out_dir = Path(args.out_dir) / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    def with_tag(p: Path, tag: str) -> Path:
        return p.with_name(f"{p.stem}_{tag}{p.suffix}")

    out_png = out_dir / with_tag(Path(args.out_png), tag).name
    out_csv = out_dir / with_tag(Path(args.out_csv), tag).name

    # Parseo de listas desde CLI
    alphas = tuple(float(x.strip()) for x in args.alphas.split(','))
    rho_vals = tuple(float(x.strip()) for x in args.rho_grid.split(','))
    r_by_alpha = {
        2.1: parse_r_sets(args.r_2_1),
        2.4: parse_r_sets(args.r_2_4),
        3.1: parse_r_sets(args.r_3_1),
    }

    # Ejecuta el experimento
    df = run_sf_fig2(
        N=args.N,
        alphas=alphas,
        r_by_alpha=r_by_alpha,
        rho_grid=rho_vals,
        p_active=args.p_active,
        k_min=args.k_min,
        max_rewire_steps=args.max_rewire_steps,  # -1 => dinámico
        rewire_factor=args.rewire_factor,        # factor × m
        tol_r=args.tol_r,
        tol_rho=args.tol_rho,
        warm_start=bool(args.warm_start),
        seed=args.seed,
        verbose=args.verbose,
    )

    # Resumen por consola (muestra primeras filas y algunos agregados)
    print(df.head())
    print(f"Filas: {len(df)}  |  N_eff(prom): {df['N_eff'].mean():.1f}  |  k_avg(prom): {df['k_avg'].mean():.2f}")

    # Salvar CSV + figura
    df.to_csv(out_csv, index=False)
    print(f"[OK] Resultados guardados en: {out_csv}")

    plot_sf_fig2(df, r_by_alpha, out_png=str(out_png))
    print(f"[OK] Figura guardada en: {out_png}")

if __name__ == '__main__':
    main()
