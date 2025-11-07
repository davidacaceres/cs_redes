# -*- coding: utf-8 -*-
"""
Fig. 3 — Ilusión de Mayoría en redes Erdős–Rényi (ER) — Paralelo por procesos (a nivel de curva)
- Un proceso = una curva (⟨k⟩, px1, r_kk), barre todos los ρ_kx dentro del proceso.
- Pool único para todo el experimento.
- Evita oversubscription: fuerza 1 hilo por proceso para BLAS (MKL/OPENBLAS/OMP).
- Logs ISO 8601, tiempos por fase, CSV y figuras en run-e2r/N{N}_{aammdd-hhmmss}/
"""

# ---- Control de threads de BLAS (¡antes de importar numpy/scipy!) ----
import os
import sys
#os.environ.setdefault("OMP_NUM_THREADS", "1")
#os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
#os.environ.setdefault("MKL_NUM_THREADS", "1")
#os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
#os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import csv
import json
import hashlib
from time import perf_counter
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from multiprocessing import Pool, cpu_count, freeze_support

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import matplotlib as mpl

# ============================ Utilidades ============================
mpl.rcParams['text.usetex'] = False
mpl.rcParams['mathtext.fontset'] = 'dejavusans'
mpl.rcParams['font.family'] = 'DejaVu Sans'
mpl.rcParams['axes.unicode_minus'] = False
TZ = ZoneInfo("America/Santiago")

def get_script_identity():
    """Devuelve (ruta_absoluta, nombre_archivo) del script en ejecución.
    Soporta ejecución como script normal o desde intérprete."""
    try:
        path = Path(__file__).resolve()  # cuando se ejecuta como script .py
    except NameError:
        path = Path(sys.argv[0]).resolve() if sys.argv and sys.argv[0] else (Path.cwd() / "<interactive>")
    return path, path.name


def iso_now(): return datetime.now(TZ).isoformat(timespec="seconds")

def make_outdir(N):
    stamp = datetime.now(TZ).strftime("%y%m%d%H%M%S")
    outdir = Path(f"run-e2r/{stamp}_N{N}"); outdir.mkdir(parents=True, exist_ok=True); return outdir

class Logger:
    def __init__(self, path): self.path = Path(path); self.path.parent.mkdir(parents=True, exist_ok=True)
    def log(self, msg):
        line = f"[{iso_now()}] {msg}"; print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as f: f.write(line + "\n")

def derive_seed(*items, base_seed=123):
    h = hashlib.sha256(); h.update(str(base_seed).encode())
    for it in items: h.update(repr(it).encode())
    return int.from_bytes(h.digest()[:4], "little")

# ============================ Núcleo experimento ============================

def generate_er_network(N, avg_degree, seed=None):
    p = avg_degree / (N - 1)
    t0 = perf_counter()
    G = nx.erdos_renyi_graph(N, p, seed=seed)
    return G, p, perf_counter() - t0

def rewire_to_target_assortativity_v2(G, target_rkk, max_iter_factor=10, tol=0.01, seed=None):
    if seed is not None: np.random.seed(seed)
    t0 = perf_counter()
    current_rkk = nx.degree_assortativity_coefficient(G)
    edges = list(G.edges()); M = len(edges)
    if M < 2: return G, current_rkk, 0, perf_counter() - t0
    max_iter = max_iter_factor * M; iters = 0
    def attempt(a,b,c,d,x1,y1,x2,y2):
        nonlocal current_rkk, edges
        if x1 == y1 or x2 == y2: return False
        if G.has_edge(x1,y1) or G.has_edge(x2,y2): return False
        G.remove_edge(a,b); G.remove_edge(c,d); G.add_edge(x1,y1); G.add_edge(x2,y2)
        new = nx.degree_assortativity_coefficient(G)
        if abs(new - target_rkk) < abs(current_rkk - target_rkk):
            current_rkk = new; edges = list(G.edges()); return True
        G.remove_edge(x1,y1); G.remove_edge(x2,y2); G.add_edge(a,b); G.add_edge(c,d); return False
    while abs(current_rkk - target_rkk) > tol and iters < max_iter:
        i1, i2 = np.random.choice(M, 2, replace=False); (a,b),(c,d) = edges[i1], edges[i2]
        if len({a,b,c,d}) != 4: iters += 1; continue
        if not attempt(a,b,c,d,a,d,c,b): attempt(a,b,c,d,a,c,b,d)
        M = len(edges); iters += 1
    return G, current_rkk, iters, perf_counter() - t0

def assign_attributes_with_rho_from_degrees(degrees, px1, target_rho_kx, max_swaps=10000, seed=None):
    if seed is not None: np.random.seed(seed)
    t0 = perf_counter()
    N = len(degrees)
    attrs = np.zeros(N, dtype=int)
    attrs[np.random.choice(N, int(round(px1*N)), replace=False)] = 1
    def corr(arr): return pearsonr(degrees, arr)[0]
    rho = corr(attrs); swaps = 0
    ones_idx = np.where(attrs==1)[0]; zeros_idx = np.where(attrs==0)[0]
    while abs(rho - target_rho_kx) > 0.01 and swaps < max_swaps and len(ones_idx)>0 and len(zeros_idx)>0:
        i1 = np.random.choice(ones_idx); i0 = np.random.choice(zeros_idx)
        attrs[i1], attrs[i0] = attrs[i0], attrs[i1]
        new_rho = corr(attrs)
        if abs(new_rho - target_rho_kx) < abs(rho - target_rho_kx):
            rho = new_rho; ones_idx = np.where(attrs==1)[0]; zeros_idx = np.where(attrs==0)[0]
        else:
            attrs[i1], attrs[i0] = attrs[i0], attrs[i1]
        swaps += 1
    return attrs, rho, swaps, perf_counter() - t0

def majority_illusion_fraction_from_neighbors(neighbors_list, attrs):
    t0 = perf_counter(); count = 0
    for nbrs in neighbors_list:
        deg = len(nbrs);
        if deg == 0: continue
        active = 0
        for n in nbrs: active += attrs[n]
        if active > deg/2: count += 1
    return count/len(neighbors_list), perf_counter() - t0

# ============================ Trabajo por proceso (CURVA) ============================

def worker_curve(task):
    """
    Ejecuta una curva completa en un proceso y devuelve:
      - logs de generación ER y rewire,
      - logs por cada ρ_kx (formateados),
      - filas CSV y serie para la figura,
      - tiempos agregados.
    """
    (N, avg_k, px1, rkk_target, rho_kxs, seed_base, rkk_tol, max_swaps, rewire_factor) = task

    t_curve0 = perf_counter()

    # (1) ER
    G, p, t_gen = generate_er_network(N, avg_k, seed=seed_base)
    L = G.number_of_edges()
    gen_log = f"Generada red ER: N={N}, <k>_obj={avg_k:.2f}, p={p:.6f}, L={L}, tiempo={t_gen:.3f}s"

    # (2) Rewire
    G, rkk_final, iters, t_rew = rewire_to_target_assortativity_v2(
        G, target_rkk=rkk_target, max_iter_factor=rewire_factor, tol=rkk_tol, seed=seed_base
    )
    rewire_log = (f"Rewire r_kk: objetivo={rkk_target:+.2f}, alcanzado={rkk_final:+.3f}, "
                  f"|Δ|={abs(rkk_final - rkk_target):.3f}, iters={iters}, tiempo={t_rew:.3f}s")

    # (3) Precompute
    degrees = np.array([deg for _, deg in G.degree()], dtype=float)
    neighbors_list = [list(G.neighbors(i)) for i in range(G.number_of_nodes())]

    # (4) Barrido rho_kx
    rows_csv = []
    series = []
    rho_logs = []
    t_attr_sum = 0.0
    t_maj_sum  = 0.0

    for rho_kx in rho_kxs:
        seed_task = derive_seed("ER", N, avg_k, px1, rkk_target, float(rho_kx), base_seed=seed_base)
        attrs, rho_final, swaps, t_attr = assign_attributes_with_rho_from_degrees(
            degrees, px1, rho_kx, max_swaps=max_swaps, seed=seed_task
        )
        frac, t_maj = majority_illusion_fraction_from_neighbors(neighbors_list, attrs)
        t_attr_sum += t_attr
        t_maj_sum  += t_maj

        # Log por rho (SIN timestamp aquí; el padre lo agregará)
        rho_logs.append(
            f"<k>={avg_k:.2f}, p={p:.6f}, px1={px1:.2f}, "
            f"r_kk_obj={rkk_target:+.2f}, r_kk_fin={rkk_final:+.3f}, "
            f"ρ_kx_obj={rho_kx:.2f}, ρ_kx_fin={rho_final:+.3f} "
            f"-> frac={frac:.3f} | t(attrs)={t_attr:.3f}s, t(frac)={t_maj:.3f}s"
        )

        rows_csv.append([
            iso_now(), N, f"{avg_k:.2f}", f"{p:.6f}", f"{px1:.2f}",
            f"{rkk_target:+.2f}", f"{rkk_final:+.6f}", f"{rho_kx:.2f}", f"{rho_final:+.6f}",
            f"{frac:.6f}", f"{t_gen:.6f}", f"{t_rew:.6f}", iters,
            f"{t_attr:.6f}", swaps, f"{t_maj:.6f}", seed_base
        ])
        series.append((float(rho_kx), float(frac)))

    t_curve = perf_counter() - t_curve0

    return {
        "avg_k": avg_k, "px1": px1, "rkk_obj": rkk_target, "rkk_fin": float(rkk_final),
        "p": float(p), "t_gen": float(t_gen), "t_rew": float(t_rew),
        "t_attr_sum": float(t_attr_sum), "t_maj_sum": float(t_maj_sum),
        "t_curve": float(t_curve),
        "gen_log": gen_log,
        "rewire_log": rewire_log,
        "rho_logs": rho_logs,
        "rows_csv": rows_csv,
        "series": sorted(series, key=lambda t: t[0])
    }


# ================================== Parámetros ==================================

N = 10000               # sube a 10000 para ver bien el uso de CPU
avg_degrees = [5.2, 2.5]
px1_values = [0.05, 0.10, 0.20]
rkk_values = [-0.5, 0.0, 0.5]
rho_kxs = np.linspace(0, 0.6, 8)

SEED = 12021974
RKK_TOL = 0.015
MAX_SWAPS = 50000
REWIRE_FACTOR = 8      # ≈ intentos por arista

NUM_PROCESOS = max(1, min(18, cpu_count() or 1))

# ================================== Main ==================================

def main():
    t0 = perf_counter()

    outdir = make_outdir(N)
    logger = Logger(outdir / "ejecucion.log")
    script_path, script_name = get_script_identity()
    logger.log(f"Ejecutando script: {script_name} | ruta={script_path}")
    logger.log(f"Inicio experimento Fig3-ER (paralelo por procesos) | carpeta={outdir}")
    logger.log(f"Procesos: usados={NUM_PROCESOS}, cpu_count={cpu_count()}")
    logger.log(f"Parámetros: N={N}, <k>={avg_degrees}, px1={px1_values}, r_kk={rkk_values}, "
               f"ρ_kx={list(np.round(rho_kxs,2))}, seed={SEED}")

    (outdir / "config.json").write_text(
        json.dumps({
            "N": N, "avg_degrees": avg_degrees, "px1_values": px1_values,
            "rkk_values": rkk_values, "rho_kxs": list(map(float, rho_kxs)),
            "seed": SEED, "processes_used": NUM_PROCESOS, "created_at": iso_now()
        }, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    csv_path = outdir / "resultados.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow([
            "timestamp_iso", "N", "avg_k_obj", "p", "px1",
            "rkk_obj", "rkk_final", "rho_kx_obj", "rho_kx_final",
            "frac_mayoria", "t_generar_red_s", "t_rewire_s", "iters_rewire",
            "t_asignar_attrs_s", "swaps_attrs", "t_mayoria_s", "seed"
        ])

    # --- construir TODAS las tareas (una por curva) ---
    tasks = []
    for avg_k in avg_degrees:
        for px1 in px1_values:
            for rkk_target in rkk_values:
                tasks.append((N, avg_k, px1, rkk_target, rho_kxs, SEED, RKK_TOL, MAX_SWAPS, REWIRE_FACTOR))

    results = {}  # results[avg_k][px1][rkk_obj] = {"rkk_final": float, "data": [(rho_kx, frac), ...]}

    # --- Pool único para todo el experimento ---
    with Pool(processes=NUM_PROCESOS) as pool:
        for out in pool.imap_unordered(worker_curve, tasks, chunksize=1):
            avg_k = out["avg_k"];
            px1 = out["px1"];
            rkk_obj = out["rkk_obj"]
            rkk_fin = out["rkk_fin"];
            p_val = out["p"]
            t_gen = out["t_gen"];
            t_rew = out["t_rew"]
            t_attr = out["t_attr_sum"];
            t_maj = out["t_maj_sum"]
            t_curve = out["t_curve"]

            # === Logs finos como antes ===
            logger.log(out["gen_log"])
            logger.log(out["rewire_log"])
            for line in out["rho_logs"]:
                logger.log(line)

            # (opcional) resumen de curva con tiempos
            logger.log(
                f"Curva lista: <k>={avg_k:.2f}, px1={px1:.2f}, r_kk_obj={rkk_obj:+.2f}, "
                f"r_kk_fin={rkk_fin:+.3f}, p={p_val:.6f}, "
                f"t_curva={t_curve:.2f}s (t_gen={t_gen:.2f}s, t_rew={t_rew:.2f}s, "
                f"t_rho_total={t_attr + t_maj:.2f}s; t_attrs={t_attr:.2f}s, t_frac={t_maj:.2f}s)"
            )

            # CSV y figura (igual que ya tenías)
            with csv_path.open("a", newline="", encoding="utf-8") as fcsv:
                writer = csv.writer(fcsv);
                writer.writerows(out["rows_csv"])
            results.setdefault(avg_k, {}).setdefault(px1, {})[rkk_obj] = {
                "rkk_final": rkk_fin, "data": out["series"]
            }

    # --- Figura final ---
    logger.log("Generando figura...")
    fig, axs = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey=True)
    for i, avg_k in enumerate(avg_degrees):
        for j, px1 in enumerate(px1_values):
            ax = axs[i, j]
            for rkk_target in rkk_values:
                series = results[avg_k][px1][rkk_target]
                rkk_fin = series["rkk_final"]
                xs = [x for x, _ in series["data"]]
                ys = [y for _, y in series["data"]]
                ax.plot(xs, ys, marker="o", linewidth=1.5,
                        label=rf"$r_{{kk}}$ obj {rkk_target:+.1f} (fin {rkk_fin:+.2f})")
            ax.set_title(f" $⟨k⟩$={avg_k}, $P(x=1)={px1}$")
            ax.set_xlabel("$K-X$ Correlación")
            ax.set_ylabel("Probabilidad de mayoria, P $\\frac{1}{2}$")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)

    plt.tight_layout()
    plt.suptitle("Ilusión de Mayoría en Redes Erdős–Rényi", y=1.02)
    outdir = csv_path.parent
    plt.savefig(outdir / "fig3_ER.png", dpi=180, bbox_inches="tight")
    plt.savefig(outdir / "fig3_ER.pdf", bbox_inches="tight")
    plt.close()

    total_time = perf_counter() - t0
    logger = Logger(outdir / "ejecucion.log")
    logger.log(f"Figuras guardadas en: {outdir/'fig3_ER.png'} y {outdir/'fig3_ER.pdf'}")
    logger.log(f"Fin experimento Fig3-ER | tiempo total={total_time:.2f}s (~{total_time/60:.2f} min) | carpeta={outdir}")

if __name__ == "__main__":
    freeze_support()
    main()
