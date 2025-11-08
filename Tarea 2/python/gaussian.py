
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fig. 5 — Aproximación Gaussiana (2 filas × 3 columnas)
Robusto a CSVs sin columna de px (SF). Si falta px en SF, usa px_sf_fijo (=0.05 por defecto).
Selecciona rkk más cercanos si no existen exactamente {-0.5, 0.0, +0.5} en SF.

Rutas:
    ER_CSV = "run-e2r/251107133431_N10000/resultados.csv"
    SF_CSV = "run/N10000_251107-014048/resultados_majority_illusion.csv"
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, poisson

# -------------------- Config --------------------
ER_CSV = "run-e2r/251107133431_N10000/resultados.csv"
SF_CSV = "run/N10000_251107-014048/resultados_majority_illusion.csv"

alpha_fijo   = 2.4
px_sf_fijo   = 0.05           # usado solo si el CSV SF no trae px
ks_er        = [5.2, 2.5]
rkk_targets  = [-0.5, 0.0, 0.5]
px_series    = [0.05, 0.10, 0.20]  # se ignorarán las que no existan en el CSV

# -------------------- Helpers --------------------
def sanitize(name:str)->str:
    return "".join(ch for ch in name.lower() if ch.isalnum())

def detect_cols(df, want):
    sane = {sanitize(c): c for c in df.columns}
    found = {}
    for key, aliases in want.items():
        col = None
        for alias in aliases:
            if alias in sane:
                col = sane[alias]; break
        found[key] = col
    return found

def p_vec(px, rho_kx, k_mean, k_sigma):
    out = px + rho_kx * (k_sigma * np.sqrt(px*(1-px))) / max(k_mean, 1e-12)
    return float(np.clip(out, 1e-9, 1-1e-9))

def p_majority_gaussian(k, p_neighbor):
    if k <= 0: return 0.0
    mu  = k * p_neighbor
    sig = np.sqrt(k * p_neighbor * (1 - p_neighbor))
    if sig == 0: return float(mu > k/2)
    z = ((k/2 + 0.5) - mu) / sig
    return float(1 - norm.cdf(z))

def er_prediction(mean_k, px, rho_grid):
    lam = mean_k
    sigma_k = np.sqrt(lam)
    k_max = int(np.ceil(lam + 6*np.sqrt(lam)))
    ks = np.arange(0, max(k_max,1)+1)
    pk = poisson.pmf(ks, lam)
    ys = []
    for rho in rho_grid:
        p_nbr = p_vec(px, rho, lam, sigma_k)
        ys.append(np.sum([p_majority_gaussian(int(k), p_nbr)*pk[i] for i,k in enumerate(ks)]))
    return np.array(rho_grid), np.array(ys)

def powerlaw_pmf(alpha, kmin=1, kmax=1000):
    ks = np.arange(kmin, kmax+1)
    w  = ks.astype(float)**(-alpha)
    pk = w/w.sum()
    return ks, pk

def sf_moments(alpha, kmin=1, kmax=1000):
    ks, pk = powerlaw_pmf(alpha, kmin, kmax)
    mean = float((ks*pk).sum())
    var  = float(((ks-mean)**2*pk).sum())
    return mean, float(np.sqrt(var)), ks, pk

def sf_prediction(alpha, px, rho_grid, kmin=1, kmax=1000):
    mean_k, sigma_k, ks, pk = sf_moments(alpha, kmin, kmax)
    ys = []
    for rho in rho_grid:
        p_nbr = p_vec(px, rho, mean_k, sigma_k)
        ys.append(np.sum([p_majority_gaussian(int(k), p_nbr)*pk[i] for i,k in enumerate(ks)]))
    return np.array(rho_grid), np.array(ys), mean_k, sigma_k

# -------------------- Carga CSV --------------------
def cargar_sf(path):
    df = pd.read_csv(path)
    cols = detect_cols(df, {
        "alpha": ["alpha"],
        "rkk":   ["rkk","r_kk","rkkobj","rkkfinal"],
        "rho":   ["rhokx","rhokxobj","rho"],
        "px":    ["px1","px","p","px_1","px01","px001","pxx","pxx1"],
        "frac":  ["fracmajorityillusion","fracmayoria","pi12","pmayor","frac"]
    })
    df = df.rename(columns={
        cols["alpha"]:"alpha",
        cols["rkk"]:"rkk",
        cols["rho"]:"rho_kx",
        cols["frac"]:"frac"
    })
    if cols["px"] is not None:
        df = df.rename(columns={cols["px"]:"px"})
    else:
        df["px"] = px_sf_fijo  # SF sin px -> asignamos fijo
    return df[["alpha","rkk","rho_kx","px","frac"]]

def cargar_er(path):
    df = pd.read_csv(path)
    cols = detect_cols(df, {
        "kavg": ["avgkobj","kavg","kmean","k_mean","avg_k_obj"],
        "rkk":  ["rkkobj","rkk","r_kk"],
        "rho":  ["rhokxobj","rhokx","rho_kx","rho"],
        "px":   ["px1","px","p"],
        "frac": ["fracmayoria","fracmajorityillusion","pi12","pmayor","frac"]
    })
    df = df.rename(columns={
        cols["kavg"]:"k_avg",
        cols["rkk"]:"rkk",
        cols["rho"]:"rho_kx",
        cols["px"]:"px",
        cols["frac"]:"frac"
    })
    return df[["k_avg","rkk","rho_kx","px","frac"]]

def nearest_values(values, targets):
    values = np.array(sorted(set(values)))
    chosen = []
    for t in targets:
        idx = np.abs(values - t).argmin()
        chosen.append(float(values[idx]))
    return chosen

# -------------------- Plot --------------------
def make_figure(er_csv=ER_CSV, sf_csv=SF_CSV, alpha=alpha_fijo):
    df_sf = cargar_sf(sf_csv)
    df_er = cargar_er(er_csv)

    # columnas rkk: usa exactas si están; si no, elige las más cercanas
    rkk_sf_cols = nearest_values(df_sf["rkk"].unique(), rkk_targets)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=True)
    # Fila superior: SF
    for j, rkk in enumerate(rkk_sf_cols):
        ax = axes[0, j]
        sub = df_sf[(df_sf["alpha"]==alpha) & (np.isclose(df_sf["rkk"], rkk))].sort_values(["px","rho_kx"])
        # qué px existen en SF (puede ser solo uno)
        pxs = sorted(sub["px"].unique())
        for px in [p for p in px_series if any(np.isclose(p, pxs))] or pxs:
            dpx = sub[np.isclose(sub["px"], px)]
            if dpx.empty: continue
            xs = dpx["rho_kx"].values; ys = dpx["frac"].values
            ax.plot(xs, ys, "o", label=f"Empírico SF  P={px:.2f}")
            xs_t, ys_t, kmean, ksig = sf_prediction(alpha, px, sorted(np.unique(xs)))
            ax.plot(xs_t, ys_t, "--", label=f"Gauss SF  P={px:.2f}")
        ax.set_title(f"Scale-free α={alpha}  rₖₖ≈{rkk:+.2f}")
        ax.set_xlabel("ρₖₓ")
        ax.set_ylabel("Fracción en mayoría (>50%)")
        ax.grid(alpha=0.25, linestyle=":")

    # Fila inferior: ER (dos k por panel)
    for j, rkk in enumerate(rkk_targets):
        ax = axes[1, j]
        sub = df_er[np.isclose(df_er["rkk"], rkk)]
        for px in [p for p in px_series if any(np.isclose(p, sub['px'].unique()))]:
            for k in ks_er:
                d = sub[np.isclose(sub["px"], px) & np.isclose(sub["k_avg"], k)].sort_values("rho_kx")
                if d.empty: continue
                xs = d["rho_kx"].values; ys = d["frac"].values
                mk = "o" if k==ks_er[0] else "s"
                ax.plot(xs, ys, mk, label=f"Empírico ER  P={px:.2f}, ⟨k⟩={k}")
                xs_t, ys_t = er_prediction(k, px, sorted(np.unique(xs)))
                ls = "--" if k==ks_er[0] else ":"
                ax.plot(xs_t, ys_t, ls, label=f"Gauss ER  P={px:.2f}, ⟨k⟩={k}")
        ax.set_title(f"ER  rₖₖ={rkk:+.2f}")
        ax.set_xlabel("ρₖₓ")
        ax.set_ylabel("Fracción en mayoría (>50%)")
        ax.grid(alpha=0.25, linestyle=":")

    # Leyenda global
    handles, labels = [], []
    for ax in axes.ravel():
        h,l = ax.get_legend_handles_labels()
        handles += h; labels += l
    # quitar duplicados
    seen=set(); H=[]; L=[]
    for h,l in zip(handles, labels):
        if l not in seen:
            H.append(h); L.append(l); seen.add(l)
    fig.legend(H, L, loc="upper center", ncol=4, frameon=False, fontsize=9)

    fig.suptitle("Fig. 5 — Aproximación Gaussiana\n(símbolos=empírico, líneas punteadas=teoría)\nFila sup.: scale-free (α fijo).  Fila inf.: ER (⟨k⟩=5.2 y 2.5).", y=1.03)
    fig.tight_layout(rect=[0,0,1,0.94])
    out = Path("run_gaus/fig5_gaussian_2x3.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    print(f"Guardado: {out}")

if __name__ == "__main__":
    make_figure()
