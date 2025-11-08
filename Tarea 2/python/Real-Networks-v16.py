# -*- coding: utf-8 -*-
"""
Fig. 4 — Ilusión de Mayoría en redes reales (local) — Paralelo por procesos (curvas)
- Pasa rutas locales por CLI (ej.: --hepth data/ca-HepTh.txt --enron data/email-Enron.txt ...)
- Un proceso = una curva (red, P(x=1)); barre todos los ρ_kx dentro del proceso.
- Logs ISO 8601 (incluye nombre del .py), CSV y figura en run-real/N0_{aammdd-hhmmss}/
"""
# ---- Limitar hilos BLAS (antes de importar numpy/scipy) ----
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
import csv as _csv
import networkx as nx

import sys
import gzip
import csv
import json
import time
import argparse
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

# ============================ Utilidades ============================

TZ = ZoneInfo("America/Santiago")
def iso_now(): return datetime.now(TZ).isoformat(timespec="seconds")

def make_outdir():
    stamp = datetime.now(TZ).strftime("%y%m%d-%H%M%S")
    outdir = Path(f"run-real/{stamp}"); outdir.mkdir(parents=True, exist_ok=True); return outdir

class Logger:
    def __init__(self, path): self.path = Path(path); self.path.parent.mkdir(parents=True, exist_ok=True)
    def log(self, msg):
        line = f"[{iso_now()}] {msg}"; print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as f: f.write(line + "\n")

def get_script_identity():
    try:
        path = Path(__file__).resolve()
    except NameError:
        path = Path(sys.argv[0]).resolve() if sys.argv and sys.argv[0] else (Path.cwd() / "<interactive>")
    return path, path.name

def derive_seed(*items, base_seed=123):
    h = hashlib.sha256(); h.update(str(base_seed).encode())
    for it in items: h.update(repr(it).encode())
    return int.from_bytes(h.digest()[:4], "little")

# ============================ Lectura de redes ============================
def load_edge_csv(path, directed=False, source_col="source", target_col="target"):
    """Lee CSV con columnas 'source','target' (opcional 'weight')."""
    G = nx.DiGraph() if directed else nx.Graph()
    with open(path, "r", encoding="utf-8") as f:
        rdr = _csv.DictReader(f)
        for row in rdr:
            u = row[source_col]; v = row[target_col]
            if not u or not v or u == v:
                continue
            G.add_edge(u, v)  # el peso es opcional; para Fig.4 no se usa
    return G

def _open_path(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No existe el archivo: {p}")
    if p.suffix == ".gz":
        return gzip.open(p, "rt", encoding="utf-8", errors="ignore")
    return p.open("r", encoding="utf-8", errors="ignore")

def load_edge_list(path, directed=False, sep=None, comments="#"):
    """Lee edge list local. Autodetecta separador si sep=None. Soporta .gz."""
    with _open_path(path) as f:
        # autodetección simple del separador
        if sep is None:
            first = ""
            for ln in f:
                if ln and not ln.startswith(comments):
                    first = ln; break
            sep = " "
            for s in (" ", "\t", ","):
                if len(first.strip().split(s)) >= 2:
                    sep = s; break
            f.seek(0)
        G = nx.DiGraph() if directed else nx.Graph()
        for line in f:
            if not line or line.startswith(comments): continue
            toks = line.strip().split(sep)
            if len(toks) < 2: continue
            u, v = toks[0], toks[1]
            if u == v: continue
            G.add_edge(u, v)
    return G

def to_undirected_mutual(G_directed: nx.DiGraph):
    """Convierte a grafo no dirigido conservando SOLO enlaces mutuos."""
    H = nx.Graph()
    for u, v in G_directed.edges():
        if u != v and G_directed.has_edge(v, u):
            H.add_edge(u, v)
    return H

def largest_cc_simple(Gu: nx.Graph):
    """GCC y sin self-loops, con relabel a [0..N-1]."""
    if Gu.number_of_nodes() == 0: return Gu
    cc = max(nx.connected_components(Gu), key=len)
    H = Gu.subgraph(cc).copy()
    H.remove_edges_from(nx.selfloop_edges(H))
    mapping = {node: i for i, node in enumerate(H.nodes())}
    return nx.relabel_nodes(H, mapping, copy=True)

# ============================ Núcleo Fig4 ============================

def compute_degrees_neighbors(G):
    degrees = np.array([deg for _, deg in G.degree()], dtype=float)
    neighbors = [list(G.neighbors(i)) for i in range(G.number_of_nodes())]
    return degrees, neighbors

def assign_attributes_with_rho_from_degrees(degrees, px1, target_rho_kx, max_swaps=10000, seed=None):
    if seed is not None: np.random.seed(seed)
    t0 = perf_counter()
    N = len(degrees)
    attrs = np.zeros(N, dtype=int)
    attrs[np.random.choice(N, int(round(px1 * N)), replace=False)] = 1

    def corr(arr):
        # Evitar ValueError de pearsonr y divisiones por cero
        if N < 2:
            return 0.0
        a = degrees.astype(float)
        b = arr.astype(float)
        va = a.var()
        vb = b.var()
        if va == 0.0 or vb == 0.0:
            return 0.0
        # Usar pearsonr solo cuando es seguro
        return float(pearsonr(a, b)[0])


    #def corr(arr): return pearsonr(degrees, arr)[0]
    rho = corr(attrs); swaps = 0
    ones_idx = np.where(attrs == 1)[0]; zeros_idx = np.where(attrs == 0)[0]
    while abs(rho - target_rho_kx) > 0.01 and swaps < max_swaps and len(ones_idx)>0 and len(zeros_idx)>0:
        i1 = np.random.choice(ones_idx); i0 = np.random.choice(zeros_idx)
        attrs[i1], attrs[i0] = attrs[i0], attrs[i1]
        new_rho = corr(attrs)
        if abs(new_rho - target_rho_kx) < abs(rho - target_rho_kx):
            rho = new_rho
            ones_idx = np.where(attrs == 1)[0]; zeros_idx = np.where(attrs == 0)[0]
        else:
            attrs[i1], attrs[i0] = attrs[i0], attrs[i1]
        swaps += 1
    return attrs, rho, swaps, perf_counter() - t0

def majority_illusion_fraction(neighbors_list, attrs):
    t0 = perf_counter(); count = 0
    for nbrs in neighbors_list:
        d = len(nbrs)
        if d == 0: continue
        active = 0
        for n in nbrs: active += attrs[n]
        if active > d/2: count += 1
    return count / len(neighbors_list), perf_counter() - t0

# ============================ Trabajo por proceso (curva) ============================

def worker_curve(task):
    """
    Curva = (name, path, directed, mutual_only, px1, rho_grid, seed, max_swaps).
    Devuelve logs por rho, filas CSV y serie.
    """
    (name, path, directed, mutual_only, px1, rho_grid, seed_base, max_swaps) = task

    # (1) Carga / limpieza
    t_load0 = perf_counter()
    # G_in = load_edge_list(path, directed=directed, sep=None, comments="#")
    if name.endswith("CSV"):
        G_in = load_edge_csv(path, directed=False)
    else:
        G_in = load_edge_list(path, directed=directed, sep=None, comments="#")
    if directed:
        G = to_undirected_mutual(G_in) if mutual_only else nx.Graph(G_in)
    else:
        G = G_in

    G = largest_cc_simple(G)
    t_load = perf_counter() - t_load0

    N = G.number_of_nodes(); L = G.number_of_edges(); rkk = nx.degree_assortativity_coefficient(G)

    # Normalizar r_kk si sale NaN por varianzas cero
    if not np.isfinite(rkk):
        rkk = 0.0

    # Si la red quedó degenerada, la omitimos con logs legibles
    if N < 2 or L == 0:
        return {
            "name": name, "N": N, "L": L, "rkk": rkk, "px1": px1,
            "rows_csv": [],
            "rho_logs": [f"red={name} OMITIDA: N={N}, L={L} (después de mutuos/GCC)"],
            "series": [],
            "t_load": float(t_load), "t_pre": 0.0,
            "t_attr_sum": 0.0, "t_maj_sum": 0.0
        }

    # (2) Precompute
    t_pre0 = perf_counter()
    degrees, neighbors = compute_degrees_neighbors(G)
    t_pre = perf_counter() - t_pre0

    # (3) Barrido de rho_kx
    rows_csv, rho_logs, series = [], [], []
    t_attr_sum = 0.0; t_maj_sum = 0.0
    for rho_kx in rho_grid:
        seed_task = derive_seed("REAL", name, px1, float(rho_kx), base_seed=seed_base)
        attrs, rho_final, swaps, t_attr = assign_attributes_with_rho_from_degrees(
            degrees, px1, rho_kx, max_swaps=max_swaps, seed=seed_task
        )
        frac, t_maj = majority_illusion_fraction(neighbors, attrs)
        t_attr_sum += t_attr; t_maj_sum += t_maj

        rho_logs.append(
            f"red={name}, N={N}, L={L}, px1={px1:.2f}, r_kk={rkk:+.3f}, "
            f"ρ_kx_obj={rho_kx:.2f}, ρ_kx_fin={rho_final:+.3f} -> frac={frac:.3f} "
            f"| t(attrs)={t_attr:.3f}s, t(frac)={t_maj:.3f}s"
        )
        rows_csv.append([
            iso_now(), name, N, L, f"{rkk:+.6f}", f"{px1:.2f}",
            f"{rho_kx:.2f}", f"{rho_final:+.6f}", f"{frac:.6f}",
            f"{t_load:.6f}", f"{t_pre:.6f}", f"{t_attr:.6f}", f"{t_maj:.6f}", seed_base
        ])
        series.append((float(rho_kx), float(frac)))

    return {
        "name": name, "N": N, "L": L, "rkk": float(rkk), "px1": px1,
        "rows_csv": rows_csv, "rho_logs": rho_logs,
        "series": sorted(series, key=lambda t: t[0]),
        "t_load": float(t_load), "t_pre": float(t_pre),
        "t_attr_sum": float(t_attr_sum), "t_maj_sum": float(t_maj_sum)
    }

# ============================ CLI ============================

def parse_rho_grid(s):
    """'a:b:n' => linspace(a,b,n) o lista '0,0.1,0.2'."""
    s = s.strip()
    if ":" in s:
        a, b, n = s.split(":")
        return np.linspace(float(a), float(b), int(n))
    return np.array([float(x) for x in s.split(",")])

def parse_px1(s):
    return [float(x) for x in s.split(",")]

def build_tasks(args):
    # Mapa: nombre -> (path, directed, mutual_only)
    nets = []
    if args.hepth:   nets.append(("HepTh",   args.hepth,   False, False))
    if args.reactome:nets.append(("Reactome",args.reactome,False, False))
    if args.digg:    nets.append(("Digg",    args.digg,    True,  False ))
    if args.twitter: nets.append(("Twitter", args.twitter, True,  False))
    if args.enron:   nets.append(("Enron",   args.enron,   False, False))
    if args.enron_csv: nets.append(("EnronCSV", args.enron_csv, False, False))
    if args.blogs:   nets.append(("Blogs",   args.blogs,   True, False))
    # Tareas: (name, path, directed, mutual_only, px1, rho_grid, seed, max_swaps)
    tasks = []
    for (name, path, directed, mutual_only) in nets:
        for px1 in args.px1:
            tasks.append((name, path, directed, mutual_only, px1, args.rho, args.seed, args.max_swaps))
    return tasks

def get_args():
    ap = argparse.ArgumentParser(description="Fig.4 — Ilusión de Mayoría en redes reales (paths locales)")

    ap.add_argument("--hepth",   type=str, default="real/data-prep/hepth/edges_all.txt", help="Ruta edge list HepTh (undirected)")
    ap.add_argument("--reactome",type=str, default="real/data-prep/reactome/edges_all.txt", help="Ruta edge list Reactome PPI (undirected)")
    ap.add_argument("--digg",    type=str, default="real/data-prep/digg/edges_mutual.txt", help="Ruta edge list Digg (directed followers)")
    ap.add_argument("--twitter", type=str, default="real/data-prep/twitter-higgs/edges_mutual.txt", help="Ruta edge list Twitter (directed followers)")
    ap.add_argument("--enron",   type=str, default="real/data-prep/enron/edges_all.txt", help="email-Enron.txt.gz Ruta edge list Enron (undirected)")
    ap.add_argument("--blogs",   type=str, default="real/data-prep/polblogs/edges_all.txt", help="Ruta edge list Political Blogs (directed)")

    ap.add_argument("--enron-csv", type=str, default="" , help="Ruta CSV Enron (source,target[,weight])")
    ap.add_argument("--mutual-twitter", dest="mutual_twitter", action="store_true",  help="Usar SOLO mutuos en Twitter (default: True)")
    ap.add_argument("--no-mutual-twitter", dest="mutual_twitter", default=True, action="store_false", help="No exigir mutuos en Twitter")
    ap.set_defaults(mutual_digg=True, mutual_twitter=True)

    ap.add_argument("--rho", type=parse_rho_grid, default="0:0.6:10",                    help="Grid de rho_kx: 'a:b:n' (linspace) o lista '0,0.1,0.2'")
    ap.add_argument("--px1", type=parse_px1, default=[0.05,0.10,0.20,0.30],                    help="Lista de prevalencias, ej: 0.05,0.1,0.2,0.3")
    ap.add_argument("--max-swaps", type=int, default=10000, help="Máximo de swaps por ajuste de rho")
    ap.add_argument("--seed", type=int, default=12021974, help="Semilla base")
    ap.add_argument("--processes", type=int, default=18, help="N° de procesos (default: cpu_count)")
    return ap.parse_args()

# ============================ Main ============================

def main():
    args = get_args()
    tasks = build_tasks(args)
    if not tasks:
        print("No se entregaron rutas. Ejemplo:\n"
              "  python fig4_real.py --hepth data/ca-HepTh.txt --enron data/email-Enron.txt "
              "--blogs data/polblogs.txt --digg data/digg_edges.txt --twitter data/twitter_edges.txt "
              "--reactome data/reactome_hs.txt --processes 12", flush=True)
        return

    outdir = make_outdir()
    logger = Logger(outdir / "ejecucion.log")
    script_path, script_name = get_script_identity()
    logger.log(f"Ejecutando script: {script_name} | ruta={script_path}")
    logger.log(f"Inicio experimento Fig4-REAL (paralelo por procesos) | carpeta={outdir}")
    logger.log(f"Procesos: usados={min(args.processes, cpu_count())}, cpu_count={cpu_count()}")
    logger.log(f"Parámetros: PX1={args.px1}, ρ_kx_grid={list(np.round(args.rho,2))}, seed={args.seed}")

    (outdir / "config.json").write_text(
        json.dumps({
            "rho_grid": list(map(float, args.rho)),
            "px1_values": args.px1,
            "seed": args.seed,
            "processes_used": min(args.processes, cpu_count()),
            "created_at": iso_now(),
            "networks": {n: p for (n,p,_,_) in [(t[0],t[1],t[2],t[3]) for t in tasks]}
        }, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    csv_path = outdir / "resultados_fig4.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow([
            "timestamp_iso", "red", "N", "L", "rkk",
            "px1", "rho_kx_obj", "rho_kx_fin",
            "frac_mayoria", "t_cargar_s", "t_precompute_s",
            "t_asignar_attrs_s", "t_mayoria_s", "seed"
        ])

    t0 = perf_counter()
    results = {}
    # Ejecutar pool único
    with Pool(processes=min(args.processes, cpu_count())) as pool:
        for out in pool.imap_unordered(worker_curve, tasks, chunksize=1):
            name = out["name"]; N = out["N"]; L = out["L"]; rkk = out["rkk"]; px1 = out["px1"]
            t_load = out["t_load"]; t_pre = out["t_pre"]
            # Logs “clásicos”
            logger.log(f"Generada red REAL: {name}, N={N}, L={L}, r_kk={rkk:+.3f}, tiempo_carga={t_load:.3f}s, tiempo_pre={t_pre:.3f}s")
            for line in out["rho_logs"]:
                logger.log(line)
            # CSV
            with csv_path.open("a", newline="", encoding="utf-8") as fcsv:
                writer = csv.writer(fcsv); writer.writerows(out["rows_csv"])
            # Figura
            entry = results.setdefault(name, {"rkk": rkk, "series_por_px1": {}})
            entry["series_por_px1"][px1] = out["series"]

    # Figura final (ordenada por r_kk desc)
    redes_ordenadas = sorted(results.items(), key=lambda kv: kv[1]["rkk"], reverse=True)
    fig, axs = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey=True)
    markers = {0.30: "o", 0.20: "^", 0.10: "s", 0.05: "x"}
    colors  = {0.30: "C0", 0.20: "C3", 0.10: "C2", 0.05: "k"}
    for idx, (name, data) in enumerate(redes_ordenadas[:6]):
        i, j = divmod(idx, 3); ax = axs[i, j]; rkk = data["rkk"]
        for px1 in sorted(data["series_por_px1"].keys(), reverse=True):
            series = data["series_por_px1"][px1]
            xs = [x for x, _ in series]; ys = [y for _, y in series]
            ax.plot(xs, ys, marker=markers.get(px1, "o"), linewidth=1.5,
                    color=colors.get(px1, None), label=rf"$P(x=1)={px1:.2f}$")
        ax.set_title(rf"{name}  ($r_{{kk}}$={rkk:+.2f})")
        ax.grid(True, alpha=0.3)
        if i == 1: ax.set_xlabel(r"Correlación grado–atributo $\rho_{kx}$")
        if j == 0: ax.set_ylabel("Fracción nodos en mayoría activa")
        ax.legend(fontsize=9)
    plt.tight_layout()
    plt.suptitle("Ilusión de Mayoría en redes reales", y=1.02)
    fig_path_png = outdir / "fig4_real.png"
    fig_path_pdf = outdir / "fig4_real.pdf"
    plt.savefig(fig_path_png, dpi=180, bbox_inches="tight")
    plt.savefig(fig_path_pdf, bbox_inches="tight")
    plt.close()

    total_time = perf_counter() - t0
    logger.log(f"Figuras guardadas en: {fig_path_png} y {fig_path_pdf}")
    logger.log(f"Fin experimento Fig4-REAL | tiempo total={total_time:.2f}s (~{total_time/60:.2f} min) | carpeta={outdir}")

if __name__ == "__main__":
    freeze_support()
    main()
