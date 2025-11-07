#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Majority Illusion en redes scale-free (script standalone)
# --------------------------------------------------------
# Este script transforma el código de notebook para ejecutarse
# como programa de línea de comando en Python.
#
# Salida solicitada (nuevo):
# - Los resultados SIEMPRE se guardan dentro de la carpeta "run/".
# - En "run/" se crea una subcarpeta con el patrón: N{N}_{aammdd-hhmmss}
#   Ejemplo: run/N500_251107-143012/
# - Dentro se guardan:
#   - resultados_majority_illusion.csv
#   - majority_illusion_plot.png
#
# ¿Qué hace?
# - Genera redes scale-free con modelo de configuración basado en Zipf.
# - Ajusta la asortatividad por grado (r_kk) mediante edge-rewiring fuerte.
# - Asigna atributos binarios con correlación objetivo rho_{kx} vía bisección.
# - Calcula la fracción de "ilusión de mayoría".
# - Ejecuta barridos sobre alpha, r_kk y rho_{kx} en paralelo.
#
# Uso básico:
#   python majority_illusion_experiment.py
#
# Opciones principales:
#   --N, --px1, --rho-min, --rho-max, --rho-steps, --max-rewire-factor, --tol, --processes
#
# Dependencias:
#   numpy, networkx, scipy, matplotlib
#
# Nota: Incluye protección if __name__ == "__main__" para compatibilidad Windows.
# --------------------------------------------------------

import argparse
import copy
import csv
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # backend no interactivo para scripts
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.stats import pearsonr
from multiprocessing import Pool, cpu_count, freeze_support, get_start_method
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
# ------------- Utilidades -------------

def timestamp():
    return time.strftime("%Y-%m-%d %H:%M:%S")

def ts_compacto():
    # Formato requerido: aammdd-hhmmss (año 2 dígitos)
    return time.strftime("%y%m%d-%H%M%S")

def seed_everything(seed: int):
    np.random.seed(seed)

# ------------- Generación de red -------------

def generate_scale_free_network(N, alpha, seed=None):
    """
    Genera una red 'scale-free' a través de una secuencia de grados Zipf(alpha),
    emparejada con modelo de configuración. Se eliminan lazos y múltiples aristas.
    """
    print(f'[{timestamp()}][alpha={alpha}] Generando red scale-free N={N}')
    start = time.time()
    rng = np.random.default_rng(seed)
    k_min, k_max = 1, int(np.sqrt(N))
    degrees = []
    # Rechazo simple hasta completar N grados en [k_min, k_max]
    while len(degrees) < N:
        k = rng.zipf(alpha)
        if k_min <= k <= k_max:
            degrees.append(int(k))
    # Asegurar suma par
    if sum(degrees) % 2 == 1:
        degrees[0] += 1
    G = nx.configuration_model(degrees, seed=seed)
    G = nx.Graph(G)  # simplificar a simple-graph
    G.remove_edges_from(nx.selfloop_edges(G))
    print(f'[{timestamp()}][alpha={alpha}] Red creada en {time.time()-start:.2f}s '
          f'| N={G.number_of_nodes()} M={G.number_of_edges()}')
    return G

# ------------- Rewiring fuerte para asortatividad -------------

def strong_rewire(G, target_rkk, max_iter_factor=10, tol=0.01, alpha=None):
    """
    Rewiring de pares de aristas (double-edge swap dirigido) guiado por gradiente
    del coeficiente de asortatividad por grado. Acepta solo si acerca a target.
    """
    iter_count = 0
    current_rkk = nx.degree_assortativity_coefficient(G)
    print(f'[{timestamp()}][alpha={alpha}] Iniciando rewiring rkk={current_rkk:.4f} target={target_rkk}')
    start = time.time()
    M = G.number_of_edges()
    max_iter = max_iter_factor * max(1, M)
    edges = list(G.edges())
    if M < 2:
        print(f'[{timestamp()}][alpha={alpha}] Red con M<2, se omite rewiring.')
        return G
    while abs(current_rkk - target_rkk) > tol and iter_count < max_iter:
        # Selección de dos aristas distintas
        i1, i2 = np.random.choice(M, 2, replace=False)
        e1, e2 = edges[i1], edges[i2]
        if len({e1[0], e1[1], e2[0], e2[1]}) != 4:
            iter_count += 1
            continue
        u, v = e1
        x, y = e2
        # Propuestas de swap (evitar multiaristas)
        if u == y or x == v or G.has_edge(u, y) or G.has_edge(x, v):
            iter_count += 1
            continue
        # Evaluación creando copia ligera
        G_temp = G.copy()
        G_temp.remove_edge(u, v)
        G_temp.remove_edge(x, y)
        G_temp.add_edge(u, y)
        G_temp.add_edge(x, v)
        new_rkk = nx.degree_assortativity_coefficient(G_temp)
        if abs(new_rkk - target_rkk) < abs(current_rkk - target_rkk):
            # Aceptar
            G = G_temp
            current_rkk = new_rkk
            # Actualizar lista de aristas y M
            edges = list(G.edges())
            M = len(edges)
        iter_count += 1
    print(f'[{timestamp()}][alpha={alpha}] Rewiring final rkk={current_rkk:.4f} '
          f'en {iter_count} iter(s) y {time.time()-start:.2f}s')
    return G

# ------------- Asignación de atributos con rho_{kx} -------------

def bisection_attribute_assignment(G, px1, target_rho_kx, max_iter=50, tol=0.01, alpha=None, rkk=None, seed=None):
    """
    Busca, por bisección en un parámetro theta, un sesgo por grado para asignar
    atributos binarios (0/1) de forma que la correlación Pearson rho(k, x)
    se acerque a 'target_rho_kx'. Mantiene densidad de activos ≈ px1.
    """
    rng = np.random.default_rng(seed)
    print(f'[{timestamp()}][alpha={alpha}, rkk={rkk}] Ajustando atributos para rho_kx={target_rho_kx}')
    start = time.time()
    n = G.number_of_nodes()
    degrees = np.array([deg for _, deg in G.degree()], dtype=float)
    if degrees.max() == degrees.min():
        degrees_norm = np.zeros_like(degrees)
    else:
        degrees_norm = (degrees - degrees.min()) / (degrees.max() - degrees.min())
    low, high = 0.0, 1.0
    best_attrs = None
    best_corr_diff = float("inf")
    n_active = int(round(px1 * n))
    for _ in range(max_iter):
        theta = 0.5 * (low + high)
        # probabilidad proporcional a mezcla (uniforme vs. por grado)
        prob_active = (1 - theta) * np.ones(n) / n + theta * (degrees_norm + 1e-12)
        prob_active /= prob_active.sum()
        attrs = np.zeros(n, dtype=int)
        idx = rng.choice(n, n_active, p=prob_active, replace=False)
        attrs[idx] = 1
        # correlación grado-atributo
        corr = pearsonr(degrees, attrs)[0]
        corr_diff = abs(corr - target_rho_kx)
        if corr_diff < best_corr_diff:
            best_corr_diff = corr_diff
            best_attrs = attrs.copy()
        # bisección
        if corr > target_rho_kx:
            high = theta
        else:
            low = theta
        if best_corr_diff < tol:
            break
    print(f'[{timestamp()}][alpha={alpha}, rkk={rkk}] Asignación terminada en {time.time()-start:.2f}s '
          f'con corr_diff={best_corr_diff:.4f}')
    return best_attrs

# ------------- Métrica de ilusión de mayoría -------------

def majority_illusion_fraction(G, attributes, alpha=None, rkk=None, rho_kx=None):
    """
    Calcula la fracción de nodos que observan >50% de vecinos activos.
    """
    print(f'[{timestamp()}][alpha={alpha}, rkk={rkk}, rho_kx={rho_kx}] Calculando fracción espejismo')
    start = time.time()
    n = G.number_of_nodes()
    # matriz de adyacencia (sparse) y producto por vector de atributos
    adj = nx.adjacency_matrix(G, nodelist=range(n))
    attrs = np.array(attributes, dtype=int).reshape(-1, 1)
    counts = adj @ attrs
    degrees = np.array([d for _, d in G.degree()], dtype=float).reshape(-1, 1)
    illusion = (counts > (degrees / 2.0)).astype(int)
    illusion_frac = float(illusion.sum() / n)
    print(f'[{timestamp()}][alpha={alpha}, rkk={rkk}, rho_kx={rho_kx}] '
          f'Fracción: {illusion_frac:.3f}, tiempo {time.time()-start:.2f}s')
    return illusion_frac

# ------------- Experimento por combinación de parámetros -------------

def experiment(args):
    alpha, rkk_target, rho_kx, G_base, px1, max_rewire_factor, tol, seed = args
    try:
        np.random.seed(seed)
        G = copy.deepcopy(G_base)
        print(f'\n[{timestamp()}][alpha={alpha}] Inicio experimento rkk={rkk_target}, rho_kx={rho_kx}')
        G = strong_rewire(G, rkk_target, max_iter_factor=max_rewire_factor, tol=tol, alpha=alpha)
        attrs = bisection_attribute_assignment(G, px1, rho_kx, alpha=alpha, rkk=rkk_target, seed=seed + 777)
        frac = majority_illusion_fraction(G, attrs, alpha=alpha, rkk=rkk_target, rho_kx=rho_kx)
        print(f'[{timestamp()}][alpha={alpha}] Resultado rkk={rkk_target}, rho_kx={rho_kx} --> frac={frac:.3f}')
        return (alpha, rkk_target, rho_kx, frac)
    except Exception as e:
        print(f'[{timestamp()}][alpha={alpha}] ERROR en experimento ({rkk_target}, {rho_kx}): {e}')
        return (alpha, rkk_target, rho_kx, float("nan"))

# ------------- Main CLI -------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Experimentos de 'Majority Illusion' en redes scale-free (script standalone)"
    )
    parser.add_argument("--N", type=int, default=500, help="Número de nodos (default: 500)")
    parser.add_argument("--px1", type=float, default=0.05, help="Fracción de nodos activos esperada (default: 0.05)")
    parser.add_argument("--max-rewire-factor", type=int, default=5, help="Factor *M para límite de iteraciones de rewiring (default: 5)")
    parser.add_argument("--tol", type=float, default=0.035, help="Tolerancia en r_kk y rho_kx (default: 0.035)")
    parser.add_argument("--seed", type=int, default=12345, help="Semilla global (default: 12345)")
    parser.add_argument("--processes", type=int, default=None, help="Procesos en paralelo (default: cpu_count)")
    parser.add_argument("--rho-min", type=float, default=0.0, help="mínimo de rho_kx (default: 0.0)")
    parser.add_argument("--rho-max", type=float, default=0.6, help="máximo de rho_kx (default: 0.6)")
    parser.add_argument("--rho-steps", type=int, default=5, help="número de puntos entre [rho-min, rho-max] (default: 5)")
    # Se elimina --out-dir: siempre se usa 'run/N{N}_{aammdd-hhmmss}'
    # Mapa por defecto de alphas -> lista de rkk objetivo
    parser.add_argument("--use-default-rkk", action="store_true",
                        help="Usar el mapa rkk por defecto (ignora --rkk-all)")
    parser.add_argument("--rkk-all", type=str, default="2.1:-0.4,-0.1,0.1,0.3;2.4:-0.45,-0.15,0.05,0.15;3.1:-0.6,-0.25,0.05,0.25",
                        help="Mapa 'alpha:lista_de_rkk;alpha:...' (usado si no se pasa --use-default-rkk)")
    return parser.parse_args()

def parse_rkk_map(args):
    if args.use_default_rkk:
        return {
            2.1: [-0.4, -0.1, 0.1, 0.3],
            2.4: [-0.45, -0.15, 0.05, 0.15],
            3.1: [-0.6, -0.25, 0.05, 0.25],
        }
    # parsear cadena rkk-all
    mapping = {}
    blocks = [b for b in args.rkk_all.split(";") if b.strip()]
    for blk in blocks:
        k, v = blk.split(":")
        alpha = float(k.strip())
        rlist = [float(x) for x in v.split(",") if x.strip()]
        mapping[alpha] = rlist
    return mapping

def main():
    freeze_support()  # Windows safety
    args = parse_args()
    # Carpeta base 'run' y subcarpeta con N y timestamp compacto
    base_dir = Path("run").resolve()
    sub_dir = base_dir / f"N{args.N}_{ts_compacto()}"
    sub_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{timestamp()}] Salidas se guardarán en: {sub_dir}")
    print(f"[{timestamp()}] Parámetros: {vars(args)}")
    seed_everything(args.seed)
    # Conjuntos de parámetros
    params = parse_rkk_map(args)
    rho_kxs = np.linspace(args.rho_min, args.rho_max, args.rho_steps)
    # Redes base por alpha (semillas reproducibles)
    print(f'[{timestamp()}] Generando redes base...')
    base_networks = {}
    for alpha in sorted(params.keys()):
        base_networks[alpha] = generate_scale_free_network(args.N, alpha, seed=42 + int(alpha * 10))
    # Lista de experimentos
    param_list = []
    base_seed = 2025
    for alpha, rkk_list in params.items():
        G_base = base_networks[alpha]
        for rkk in rkk_list:
            for rho_kx in rho_kxs:
                param_list.append((alpha, rkk, float(rho_kx), G_base, args.px1, args.max_rewire_factor, args.tol, base_seed))
                base_seed += 1
    # Paralelización
    nproc = args.processes or cpu_count()
    print(f'[{timestamp()}] Iniciando experimentos en paralelo... (processes={nproc})')
    if get_start_method(allow_none=True) is None:
        # En algunos entornos, start_method no está configurado; Pool lo hará por defecto.
        pass
    with Pool(processes=nproc) as pool:
        results = pool.map(experiment, param_list)
    print(f'[{timestamp()}] Experimentos completados.')
    # Agregar resultados por alpha/rkk
    results_dict = {}
    for alpha, rkk, rho_kx, frac in results:
        results_dict.setdefault(alpha, {}).setdefault(rkk, []).append((rho_kx, frac))
    # Guardar CSV
    csv_path = sub_dir / "resultados_majority_illusion.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["alpha", "r_kk", "rho_kx", "frac_majority_illusion"])
        for alpha in sorted(results_dict.keys()):
            for rkk in sorted(results_dict[alpha].keys()):
                for rho_kx, frac in sorted(results_dict[alpha][rkk], key=lambda x: x[0]):
                    writer.writerow([alpha, rkk, rho_kx, frac])
    print(f"[{timestamp()}] Resultados guardados en: {csv_path}")
    # Graficar
    fig, axes = plt.subplots(1, len(sorted(params.keys())), figsize=(18, 5), sharey=True)
    if len(sorted(params.keys())) == 1:
        axes = [axes]
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
    markers = ['o', 's', '^', 'D', 'v', 'X']
    for idx, alpha in enumerate(sorted(params.keys())):
        ax = axes[idx]
        for cidx, rkk in enumerate(params[alpha]):
            data = sorted(results_dict[alpha][rkk], key=lambda x: x[0])
            x = [d[0] for d in data]
            y = [d[1] for d in data]
            linestyle = '-' if alpha == 2.1 else '--' if alpha == 2.4 else ':'
            ax.plot(x, y, label=f'$r_{{kk}}={rkk}$',
                    color=colors[cidx % len(colors)], marker=markers[cidx % len(markers)],
                    linestyle=linestyle)
        ax.set_title(f'Alpha={alpha}')
        ax.set_xlabel('Correlación grado-atributo $\rho_{kx}$')
        ax.grid(True, alpha=0.3, linestyle=':')
        if idx == 0:
            ax.set_ylabel('Fracción con mayoría de vecinos activos')
        ax.legend(loc="best", fontsize=8)
    fig.suptitle('Ilusión de Mayoría en Redes Scale-Free por Alpha', y=1.02, fontsize=14)
    try:
        fig.tight_layout()
    except Exception as e:
        print(f"[ {timestamp()} ] Aviso: tight_layout falló ({e}). Uso subplots_adjust como respaldo.")
        plt.subplots_adjust(wspace=0.3, hspace=0.3, left=0.07, right=0.98, top=0.88, bottom=0.12)
    png_path = sub_dir / "majority_illusion_plot.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"[{timestamp()}] Figura guardada en: {png_path}")
    print(f"[{timestamp()}] ¡Listo!")

if __name__ == "__main__":
    main()
