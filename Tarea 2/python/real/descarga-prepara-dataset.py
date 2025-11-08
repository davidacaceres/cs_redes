#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prep Fig.4 — Descarga y preprocesamiento de redes reales
--------------------------------------------------------
Genera edge lists listos para el experimento (Fig. 4: espejismo de mayoría)
a partir de datasets públicos o archivos locales:

Datasets soportados (banderas):
  --enron           (SNAP) email-Enron.txt.gz  [no dirigido]
  --hepth           (SNAP) ca-HepTh.txt.gz     [no dirigido]
  --twitter-higgs   (SNAP) social_network.edgelist.gz [DIRIGIDO followers → se usan MUTUOS]
  --polblogs PATH   Ruta local a polblogs.gml o edge list dirigido (colapsado a NO dirigido)
  --reactome PATH   Ruta local a PPI (dos columnas o CSV/TSV con columnas nombradas)

Salida por dataset (en ./data-prep/<nombre>/):
  - edges_all.txt         (no dirigido, colapsado)
  - edges_mutual.txt      (solo si la fuente es dirigida → mutuos A↔B; p.ej. Twitter/Digg)
  - stats.json            (N, L, N_GCC, L_GCC, r_kk, etc.)
  - ejecucion.log         (log de esta preparación)

Descargas automáticas usadas:
  - ENRON (SNAP):    https://snap.stanford.edu/data/email-Enron.txt.gz
  - ca-HepTh (SNAP): https://snap.stanford.edu/data/ca-HepTh.txt.gz
  - Twitter-Higgs:   https://snap.stanford.edu/data/higgs-twitter.html (archivo: social_network.edgelist.gz)

Referencias:
  - Enron (SNAP):     ver link de descarga directo en la página del dataset.
  - HepTh (SNAP):     ver link de descarga directo en la página del dataset.
  - Higgs Twitter:    página con archivos y ‘social_network.edgelist.gz’.
  - Political Blogs:  dataset de Adamic & Glance (dirigido); puedes obtenerlo (GML) desde mirrors confiables.
  - Reactome PPI:     descargas oficiales de PPI/MITAB en Reactome.

Requisitos:
  - Python 3.9+
  - pip install networkx requests pandas (pandas solo si parseas CSV/TSV para reactome)
"""
from __future__ import annotations
import argparse
import csv
import gzip
import io
import json
import os
import sys
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Iterable
import re
import networkx as nx
from urllib.parse import urljoin


import re, json, time, gzip, zipfile
from urllib.parse import urljoin, urlparse, parse_qs
import networkx as nx


# Evita sobre-suscripción cuando networkx usa BLAS (si existiera)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

DIGG_DOWNLOADS_INDEX = "https://www.isi.edu/people-lerman/research/downloads/"
DIGG_OLD_PAGE = "https://www.isi.edu/~lerman/downloads/digg2009.html"

BROWSER_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
    "Accept": "*/*", "Accept-Language": "en-US,en;q=0.9"
}
DIGG_FILE_RE = r'(friend|fan|follow|friendship|followers).*\.(txt|csv|tsv|gz|zip)$'

try:
    import requests
except Exception:
    requests = None



# ============================ Utilidades de logging ============================

def iso_now() -> str:
    # Tu zona horaria es America/Santiago (UTC-03), ajusta si quieres exacto
    return datetime.now().isoformat(timespec="seconds")

def log(logfile: Path, msg: str):
    line = f"[{iso_now()}] {msg}"
    print(line, flush=True)
    logfile.parent.mkdir(parents=True, exist_ok=True)
    with logfile.open("a", encoding="utf-8") as f:
        f.write(line + "\n")

def _get_text_following_403(url, timeout=180):
    r = requests.get(url, timeout=timeout, headers=BROWSER_HEADERS, allow_redirects=True)
    if r.status_code == 403 and url.startswith("https://"):
        alt = "http://" + url[len("https://"):]
        r = requests.get(alt, timeout=timeout, headers=BROWSER_HEADERS, allow_redirects=True)
    r.raise_for_status()
    return r.text, r.url

# ================================ Descargas ===================================

ENRON_URL   = "https://snap.stanford.edu/data/email-Enron.txt.gz"
HEPTH_URL   = "https://snap.stanford.edu/data/ca-HepTh.txt.gz"
HIGGS_FOLLOWERS_URL = "https://snap.stanford.edu/data/higgs-social_network.edgelist.gz"
REACTOME_HUMAN_MITAB_URL = "https://reactome.org/download/current/interactors/reactome.homo_sapiens.interactions.psi-mitab.txt"
POLBLOGS_URLS = [
    "https://websites.umich.edu/~mejn/netdata/polblogs.zip",  # Newman (oficial) :contentReference[oaicite:0]{index=0}
    "http://www-personal.umich.edu/~mejn/netdata/polblogs.zip"  # Alias legacy :contentReference[oaicite:1]{index=1}
]
# Fallbacks directos (mirrors). NRVIS/networkrepository:
DIGG_FRIENDS_DIRECT_URLS = [
    "https://nrvis.com/download/data/dynamic/digg-friends.zip",
    # puedes agregar otros mirrors aquí si quieres
]
DIGG_FRIEND_HREF_RE = r'(friend|fan|follow).*\.(txt|gz|zip)$'

def _find_digg_friend_file_from_official(rawdir, logfile):
    """
    1) Abre página 'Downloads' oficial y encuentra el botón 'Download Digg 2009 data set'.
    2) Ese botón apunta a un loader con ?src=old_page. Extraemos 'src' y la descargamos.
    3) En la old_page buscamos un link con DIGG_FILE_RE. Descargamos ese archivo a rawdir.
    Retorna Path al archivo descargado.
    """
    # Paso 1: página de Downloads
    log(logfile, f"Buscando enlace oficial en {DIGG_DOWNLOADS_INDEX} ...")
    html, _ = _get_text_following_403(DIGG_DOWNLOADS_INDEX)

    # Buscar el href del botón "Download Digg 2009 data set"
    m = re.search(r'href=[\'"]([^\'"]+)[\'"][^>]*>\s*Download\s+Digg\s+2009\s+data\s+set\s*<', html, re.I)
    if not m:
        raise RuntimeError("No se encontró el botón 'Download Digg 2009 data set' en la página oficial.")

    loader_url = m.group(1)  # suele ser https://usc-isi-i2.github.io/home/?src=...
    # Paso 2: extraer ?src= con la old_page
    parsed = urlparse(loader_url)
    qs = parse_qs(parsed.query or "")
    src_page = (qs.get("src") or [DIGG_OLD_PAGE])[0]

    # Descargar old_page (con headers + fallback)
    log(logfile, f"Abrir página histórica: {src_page}")
    old_html, final_src = _get_text_following_403(src_page)

    # Paso 3: buscar link del archivo de amistades
    # Permitimos relativos; luego hacemos urljoin.
    hrefs = re.findall(r'href=[\'"]([^\'"]+)[\'"]', old_html, flags=re.I)
    cand = next((h for h in hrefs if re.search(DIGG_FILE_RE, h, re.I)), None)
    if not cand:
        raise RuntimeError("No se encontró enlace de amistades en la página oficial.")

    file_url = urljoin(final_src, cand)
    fname = file_url.split("/")[-1]
    dest = rawdir / fname
    http_get(file_url, dest, logfile, headers=BROWSER_HEADERS)
    return dest

def _extract_edgefile_from_zip(zpath, logfile):
    import zipfile, re
    with zipfile.ZipFile(zpath, "r") as zf:
        # 1) prioriza nombres con friend/fan/follow/friendship
        pri = [m for m in zf.namelist() if re.search(r'(friend|fan|follow|friendship)', m, re.I)]
        # 2) si no, busca extensiones típicas de edgelist
        if not pri:
            pri = [m for m in zf.namelist() if re.search(r'\.(txt|csv|tsv|edges)$', m, re.I)]
        if not pri:
            raise RuntimeError("ZIP sin archivo de aristas detectable.")
        member = pri[0]
        out = zpath.parent / Path(member).name
        with zf.open(member) as src, open(out, "wb") as dst:
            dst.write(src.read())
        log(logfile, f"Extraído {member} -> {out}")
        return out



def _extract_friend_like_from_zip(zpath, logfile):
    with zipfile.ZipFile(zpath, "r") as zf:
        members = [m for m in zf.namelist() if re.search(r'(friend|fan|follow|friendship)', m, re.I)]
        if not members:
            raise RuntimeError("ZIP de Digg sin archivo de amistades detectable.")
        member = members[0]
        out = zpath.parent / Path(member).name
        with zf.open(member) as src, open(out, "wb") as dst:
            dst.write(src.read())
        log(logfile, f"Extraído {member} -> {out}")
        return out

def _read_pairs_two_cols(path, directed=True):
    G = nx.DiGraph() if directed else nx.Graph()
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line or line.startswith("#"): continue
            parts = re.split(r"[,\s]+", line.strip())
            if len(parts) >= 2 and parts[0] and parts[1] and parts[0] != parts[1]:
                G.add_edge(parts[0], parts[1])
    return G


def http_get_first_matching_from_page(page_url: str, rawdir: Path, logfile: Path, href_regex=DIGG_FRIEND_HREF_RE) -> Path:
    if requests is None:
        raise RuntimeError("El módulo 'requests' es requerido para descargar Digg 2009.")
    log(logfile, f"Explorando {page_url} ...")
    r = requests.get(page_url, timeout=180, headers=BROWSER_HEADERS, allow_redirects=True)
    if r.status_code == 403 and page_url.startswith("https://"):
        alt = "http://" + page_url[len("https://"):]
        r = requests.get(alt, timeout=180, headers=BROWSER_HEADERS, allow_redirects=True)
    r.raise_for_status()
    hrefs = re.findall(r'href=[\'"]([^\'"]+)[\'"]', r.text, flags=re.I)
    cand = next((h for h in hrefs if re.search(href_regex, h, flags=re.I)), None)
    if not cand:
        raise RuntimeError("No se encontró enlace de amistades en la página oficial.")
    url = urljoin(page_url, cand)
    fname = url.split("/")[-1]
    dest = rawdir / fname
    http_get(url, dest, logfile)
    return dest

def http_get(url, dest, logfile, headers=None, chunk=1<<20):
    if requests is None:
        raise RuntimeError("El módulo 'requests' no está disponible. Instálalo con: pip install requests")
    hdrs = headers or BROWSER_HEADERS
    t0 = time.perf_counter()
    log(logfile, f"Descargando: {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, timeout=600, headers=hdrs, allow_redirects=True, stream=True) as r:
        # fallback a http si 403 con https
        if r.status_code == 403 and url.startswith("https://"):
            alt = "http://" + url[len("https://"):]
            r = requests.get(alt, timeout=600, headers=hdrs, allow_redirects=True, stream=True)
        r.raise_for_status()
        with open(dest, "wb") as f:
            for part in r.iter_content(chunk_size=chunk):
                if part:
                    f.write(part)
    dt = time.perf_counter() - t0
    log(logfile, f"Descarga OK → {dest} ({dest.stat().st_size} bytes) en {dt:.2f}s")




def http_get_try(urls: list[str], dest: Path, logfile: Path):
    last_err = None
    for u in urls:
        try:
            http_get(u, dest, logfile)
            return
        except Exception as e:
            last_err = e
            log(logfile, f"Fallo al descargar {u}: {e}")
    raise last_err if last_err else RuntimeError("No se pudo descargar desde ninguna URL")

def prep_polblogs_download(rawdir: Path, outdir: Path, logfile: Path) -> PrepResult:
    """
    Political Blogs (Adamic & Glance) — descarga zip oficial de Newman,
    extrae polblogs.gml y procesa como dirigido→colapsado no dirigido (Fig.4).
    """
    rawdir.mkdir(parents=True, exist_ok=True)
    zpath = rawdir / "polblogs.zip"
    gml_path = rawdir / "polblogs.gml"

    if not zpath.exists() and not gml_path.exists():
        http_get_try(POLBLOGS_URLS, zpath, logfile)

    # Extraer polblogs.gml si solo tenemos el zip
    if zpath.exists() and not gml_path.exists():
        log(logfile, f"Extrayendo GML desde {zpath} ...")
        with zipfile.ZipFile(zpath, "r") as zf:
            members = [m for m in zf.namelist() if m.lower().endswith(".gml")]
            if not members:
                raise RuntimeError("Zip sin .gml; no se puede continuar.")
            with zf.open(members[0]) as zf_gml, open(gml_path, "wb") as out:
                out.write(zf_gml.read())

    # Procesar como en la versión local
    log(logfile, f"Leyendo GML {gml_path} ...")
    Gd = read_polblogs(gml_path)
    n_raw, l_raw = Gd.number_of_nodes(), Gd.number_of_edges()

    Gu = collapse_to_undirected(Gd)
    Gu = remove_selfloops_and_multi(Gu)
    Gu = graph_gcc_simple(Gu)
    Gu = nx.convert_node_labels_to_integers(Gu)

    out_all = outdir / "edges_all.txt"
    write_edgelist_txt(Gu, out_all)
    st = stats_json(Gu)
    (outdir / "stats.json").write_text(json.dumps({
        "dataset": "Political Blogs (download Newman, colapsado no dirigido)",
        "source": str(gml_path),
        "n_raw_directed": n_raw, "l_raw_directed": l_raw,
        "n_gcc": st["N"], "l_gcc": st["L"],
        "avg_k": st["avg_k"], "r_kk": st["r_kk"]
    }, indent=2), encoding="utf-8")

    return PrepResult("PoliticalBlogs", n_raw, l_raw, st["N"], st["L"], st["r_kk"], out_all, None, 0.0)

def prep_reactome_download(rawdir: Path, outdir: Path, logfile: Path) -> PrepResult:
    """
    Descarga Reactome (humano, PSI-MITAB), lo convierte a edge list no dirigido y guarda la GCC.
    """
    gz = rawdir / "reactome.homo_sapiens.interactions.psi-mitab.txt"
    if not gz.exists():
        http_get(REACTOME_HUMAN_MITAB_URL, gz, logfile)

    G = parse_reactome_mitab_to_graph(gz, only_human=True)
    n_raw, l_raw = G.number_of_nodes(), G.number_of_edges()

    G = graph_gcc_simple(G)
    G = nx.convert_node_labels_to_integers(G)

    out_all = outdir / "edges_all.txt"
    write_edgelist_txt(G, out_all)

    st = stats_json(G)
    (outdir / "stats.json").write_text(json.dumps({
        "dataset": "Reactome PPI (human, PSI-MITAB)",
        "source": str(gz),
        "n_raw": n_raw, "l_raw": l_raw,
        "n_gcc": st["N"], "l_gcc": st["L"],
        "avg_k": st["avg_k"], "r_kk": st["r_kk"]
    }, indent=2), encoding="utf-8")

    return PrepResult("Reactome", n_raw, l_raw, st["N"], st["L"], st["r_kk"], out_all, None, 0.0)

# ============================== Lectura genérica ==============================

def read_edgelist_any(path: Path, directed=False, sep=None, comments="#") -> nx.Graph:
    """Lee .txt/.gz con pares por línea. Si directed=True retorna DiGraph."""
    create = nx.DiGraph if directed else nx.Graph
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
            G = nx.read_edgelist(f, create_using=create(), delimiter=sep, comments=comments)
    else:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            G = nx.read_edgelist(f, create_using=create(), delimiter=sep, comments=comments)
    return G



def read_polblogs(path) -> nx.DiGraph:
    """
    Lee polblogs desde .gml o edge list dirigido.
    Si el GML contiene aristas duplicadas, las deduplica.
    Retorna siempre DiGraph (dirigido).
    """
    p = str(path)
    if p.lower().endswith(".gml"):
        try:
            # Usa el ID numérico de nodos si existe; evita colisiones por 'label'
            G = nx.read_gml(p, label="id", destringizer=int)
            if not isinstance(G, nx.DiGraph):
                G = nx.DiGraph(G)
            return G
        except Exception as e:
            # Fallback: parser simple de bloques GML → deduplicar aristas
            if "duplicated" not in str(e).lower():
                raise
            # Parse manual de bloques edge [ source X target Y ... ]
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            directed_flag = 1 if re.search(r"\bdirected\s+1\b", text) else 0
            edge_blocks = re.findall(r"edge\s*\[(.*?)\]", text, flags=re.DOTALL | re.IGNORECASE)
            edges = set()
            for blk in edge_blocks:
                ms = re.search(r"\bsource\s+(-?\d+)", blk)
                mt = re.search(r"\btarget\s+(-?\d+)", blk)
                if ms and mt:
                    u = int(ms.group(1)); v = int(mt.group(1))
                    if u != v:
                        edges.add((u, v))  # set => dedup
            G = nx.DiGraph() if directed_flag else nx.Graph()
            G.add_edges_from(edges)
            # Asegura DiGraph para mantener la semántica "dirigida" previa al colapso
            if not isinstance(G, nx.DiGraph):
                G = nx.DiGraph(G)
            return G
    # Si no es GML, asumimos edge list dirigido "u v" por línea
    return read_edgelist_any(Path(path), directed=True)


def read_reactome_pairs(path: Path, source_col: Optional[str]=None, target_col: Optional[str]=None) -> nx.Graph:
    """
    Lee interacciones proteína-proteína desde:
      - TSV/CSV con columnas nombradas (source_col/target_col) o
      - archivo con 2 columnas (tab/espacio/coma) sin encabezado.
    Devuelve Graph no dirigido.
    """
    # Intento con pandas si hay encabezados; si falla, edge list plano
    try:
        import pandas as pd
        df = pd.read_csv(path, sep=None, engine="python")
        if source_col and target_col and source_col in df.columns and target_col in df.columns:
            pairs = df[[source_col, target_col]].dropna().astype(str).values.tolist()
        else:
            # Si tiene al menos 2 columnas, tomo las 2 primeras
            cols = list(df.columns)
            if len(cols) < 2:
                raise ValueError("Archivo con menos de 2 columnas.")
            pairs = df[[cols[0], cols[1]]].dropna().astype(str).values.tolist()
        G = nx.Graph()
        G.add_edges_from((str(u), str(v)) for u, v in pairs if u != v)
        return G
    except Exception:
        # Fallback: tratar como edge list sin encabezado
        return read_edgelist_any(path, directed=False, sep=None)

# ============================ Transformaciones ================================
def parse_reactome_mitab_to_graph(path_gz: Path, only_human=True) -> nx.Graph:
    """
    Lee PSI-MITAB (gz o txt) de Reactome y retorna Graph no dirigido.
    - Extrae interactor A/B de las 2 primeras columnas.
    - Prefiere IDs 'uniprotkb:'; si no hay, usa el primer ID de la celda.
    - Filtra 'taxid:9606' en ambos interactores si only_human=True.
    - Deduplica y elimina self-loops.
    """
    def pick_id(cell: str) -> Optional[str]:
        # cada celda puede traer 'uniprotkb:P12345|reactome:XYZ|...'
        for token in cell.split("|"):
            t = token.strip()
            if t.lower().startswith("uniprotkb:"):
                return t.split(":", 1)[1]
        # fallback: primer token sin prefijo
        t0 = cell.split("|", 1)[0].strip()
        return t0.split(":", 1)[-1] if ":" in t0 else t0

    G = nx.Graph()
    opener = gzip.open if path_gz.suffix == ".gz" else open
    with opener(path_gz, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 2:
                continue
            # filtro humano: exige 'taxid:9606' al menos dos veces en la línea (ambos interactores)
            if only_human and line.count("taxid:9606") < 2:
                continue
            a = pick_id(cols[0]); b = pick_id(cols[1])
            if not a or not b or a == b:
                continue
            # descarta moléculas pequeñas si aparecen (por si vinieran en all_species)
            if a.lower().startswith("chebi") or b.lower().startswith("chebi"):
                continue
            G.add_edge(a, b)
    # limpieza mínima
    G.remove_edges_from(nx.selfloop_edges(G))
    return G

def graph_gcc_simple(G: nx.Graph) -> nx.Graph:
    """Devuelve la componente conexa gigante (copia)."""
    if G.number_of_nodes() == 0:
        return G.copy()
    if isinstance(G, nx.DiGraph):
        # para dirigidas, usamos WCC→colapsamos luego
        C = max(nx.weakly_connected_components(G), key=len)
        return G.subgraph(C).copy()
    C = max(nx.connected_components(G), key=len)
    return G.subgraph(C).copy()

def collapse_to_undirected(Gd: nx.DiGraph) -> nx.Graph:
    """Colapsa una red dirigida a no dirigida (u–v si existe u→v o v→u)."""
    Gu = nx.Graph()
    Gu.add_edges_from(Gd.edges())
    return Gu

def mutual_undirected(Gd: nx.DiGraph) -> nx.Graph:
    """Construye no dirigido con SOLO enlaces mutuos A↔B."""
    Gu = nx.Graph()
    # Truco rápido: agrega (u,v) si existe (u,v) y (v,u)
    for u, v in Gd.edges():
        if u != v and Gd.has_edge(v, u):
            if not Gu.has_edge(u, v):
                Gu.add_edge(u, v)
    return Gu

def remove_selfloops_and_multi(G: nx.Graph) -> nx.Graph:
    G = G.copy()
    G.remove_edges_from(nx.selfloop_edges(G))
    # networkx Graph ya evita multi; si viniera de MultiGraph, colapsa:
    if isinstance(G, nx.MultiGraph):
        H = nx.Graph()
        H.add_nodes_from(G.nodes())
        for u, v in G.edges():
            if u != v:
                H.add_edge(u, v)
        return H
    return G

def write_edgelist_txt(G: nx.Graph, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for u, v in G.edges():
            f.write(f"{u} {v}\n")

def stats_json(G: nx.Graph) -> dict:
    N = G.number_of_nodes()
    L = G.number_of_edges()
    avg_k = (2 * L / N) if N else 0.0
    try:
        rkk = nx.degree_assortativity_coefficient(G)
    except Exception:
        rkk = float("nan")
    return {"N": N, "L": L, "avg_k": avg_k, "r_kk": rkk}

# ================================ Pipelines ===================================

@dataclass
class PrepResult:
    name: str
    n_raw: int
    l_raw: int
    n_gcc: int
    l_gcc: int
    rkk: float
    out_all: Optional[Path]
    out_mut: Optional[Path]
    t_total_s: float

def prep_digg(rawdir: Path, outdir: Path, logfile: Path) -> PrepResult:
    rawdir.mkdir(parents=True, exist_ok=True)

    # 0) ¿ya hay archivo local?
    existing = (list(rawdir.glob("*friend*.*")) + list(rawdir.glob("*fan*.*")) +
                list(rawdir.glob("*follow*.*")) + list(rawdir.glob("*friendship*.*")) +
                list(rawdir.glob("digg*.*")))
    if existing:
        src = sorted(existing)[0]
        log(logfile, f"Usando archivo local existente: {src}")
    else:
        # 1) Intento oficial (si lo tienes implementado; si no, puedes saltarlo)
        try:
            # ... tu lógica de scraping de la página oficial si existe ...
            # si funcionara: src = archivo_descargado
            raise RuntimeError("Sin enlace directo en la página oficial (loader/JS).")
        except Exception as e:
            log(logfile, f"Oficial no disponible: {e}. Probando mirrors...")

        # 2) Fallback directo: NRVIS
        last_err = None
        for u in DIGG_FRIENDS_DIRECT_URLS:
            try:
                fname = u.rsplit("/", 1)[-1]
                cand = rawdir / fname
                http_get(u, cand, logfile, headers=BROWSER_HEADERS)
                src = cand
                break
            except Exception as ex:
                last_err = ex
                log(logfile, f"Falló {u}: {ex}")
        if 'src' not in locals():
            raise RuntimeError(f"No se pudo descargar Digg desde mirrors. Último error: {last_err}")

    # 3) Si es ZIP, extrae archivo con aristas
    if str(src).lower().endswith(".zip"):
        src = _extract_edgefile_from_zip(src, logfile)

    # 4) Dirigido -> MUTUOS A<->B -> no dirigido -> GCC -> relabel
    Gd = _read_pairs_two_cols(src, directed=True)
    n_raw, l_raw = Gd.number_of_nodes(), Gd.number_of_edges()

    Gm = mutual_undirected(Gd)         # usa tus helpers existentes
    Gm = remove_selfloops_and_multi(Gm)
    Gm = graph_gcc_simple(Gm)
    Gm = nx.convert_node_labels_to_integers(Gm)

    outdir.mkdir(parents=True, exist_ok=True)
    out_mut = outdir / "edges_mutual.txt"
    write_edgelist_txt(Gm, out_mut)
    st = stats_json(Gm)

    (outdir / "stats.json").write_text(json.dumps({
        "dataset": "Digg 2009 (amistades, MUTUOS)",
        "official_page": DIGG_OLD_PAGE,
        "download_source": str(src),   # quedará NRVIS si vino de mirror
        "n_raw_directed": n_raw, "l_raw_directed": l_raw,
        "n_gcc": st["N"], "l_gcc": st["L"], "avg_k": st["avg_k"], "r_kk": st["r_kk"]
    }, indent=2), encoding="utf-8")

    return PrepResult("Digg2009", n_raw, l_raw, st["N"], st["L"], st["r_kk"], None, out_mut, 0.0)





def prep_enron(rawdir: Path, outdir: Path, logfile: Path) -> PrepResult:
    """Enron (SNAP): no dirigido; limpieza + GCC."""
    gz = rawdir / "email-Enron.txt.gz"
    if not gz.exists():
        http_get(ENRON_URL, gz, logfile)
    G = read_edgelist_any(gz, directed=False)
    n_raw, l_raw = G.number_of_nodes(), G.number_of_edges()
    G = remove_selfloops_and_multi(G)
    G = graph_gcc_simple(G)
    G = nx.convert_node_labels_to_integers(G)  # para ficheros compactos
    out_all = outdir / "edges_all.txt"
    write_edgelist_txt(G, out_all)
    st = stats_json(G)
    # Guardar stats
    (outdir / "stats.json").write_text(json.dumps({
        "dataset": "Enron (SNAP)",
        "n_raw": n_raw, "l_raw": l_raw,
        "n_gcc": st["N"], "l_gcc": st["L"],
        "avg_k": st["avg_k"], "r_kk": st["r_kk"]
    }, indent=2), encoding="utf-8")
    return PrepResult("Enron", n_raw, l_raw, st["N"], st["L"], st["r_kk"], out_all, None, 0.0)

def prep_hepth(rawdir: Path, outdir: Path, logfile: Path) -> PrepResult:
    """ca-HepTh (SNAP): no dirigido; limpieza + GCC."""
    gz = rawdir / "ca-HepTh.txt.gz"
    if not gz.exists():
        http_get(HEPTH_URL, gz, logfile)
    G = read_edgelist_any(gz, directed=False)
    n_raw, l_raw = G.number_of_nodes(), G.number_of_edges()
    G = remove_selfloops_and_multi(G)
    G = graph_gcc_simple(G)
    G = nx.convert_node_labels_to_integers(G)
    out_all = outdir / "edges_all.txt"
    write_edgelist_txt(G, out_all)
    st = stats_json(G)
    (outdir / "stats.json").write_text(json.dumps({
        "dataset": "ca-HepTh (SNAP)",
        "n_raw": n_raw, "l_raw": l_raw,
        "n_gcc": st["N"], "l_gcc": st["L"],
        "avg_k": st["avg_k"], "r_kk": st["r_kk"]
    }, indent=2), encoding="utf-8")
    return PrepResult("HepTh", n_raw, l_raw, st["N"], st["L"], st["r_kk"], out_all, None, 0.0)

def prep_twitter_higgs(rawdir: Path, outdir: Path, logfile: Path) -> PrepResult:
    """
    Twitter–Higgs followers (SNAP): dirigido → construimos NO dirigido de MUTUOS,
    luego GCC. Archivo: social_network.edgelist.gz (followers graph).
    """
    gz = rawdir / "higgs-social_network.edgelist.gz"
    if not gz.exists():
        try:
            http_get(HIGGS_FOLLOWERS_URL, gz, logfile)
        except Exception as e:
            log(logfile, f"ERROR descargando Twitter–Higgs: {e}")
            raise

    Gd = read_edgelist_any(gz, directed=True)
    n_raw, l_raw = Gd.number_of_nodes(), Gd.number_of_edges()
    Gm = mutual_undirected(Gd)            # SOLO mutuos A↔B
    Gm = remove_selfloops_and_multi(Gm)
    Gm = graph_gcc_simple(Gm)
    Gm = nx.convert_node_labels_to_integers(Gm)

    out_mut = outdir / "edges_mutual.txt"
    write_edgelist_txt(Gm, out_mut)
    st = stats_json(Gm)
    (outdir / "stats.json").write_text(json.dumps({
        "dataset": "Twitter–Higgs (SNAP) followers (MUTUOS)",
        "n_raw_directed": n_raw, "l_raw_directed": l_raw,
        "n_gcc": st["N"], "l_gcc": st["L"],
        "avg_k": st["avg_k"], "r_kk": st["r_kk"]
    }, indent=2), encoding="utf-8")
    return PrepResult("Twitter-Higgs", n_raw, l_raw, st["N"], st["L"], st["r_kk"], None, out_mut, 0.0)

def prep_polblogs_local(path: Path, outdir: Path, logfile: Path) -> PrepResult:
    """
    Political Blogs (Adamic & Glance): DIRIGIDO → para Fig.4 se usa NO dirigido
    por colapso (u–v si existe u→v o v→u). NO se restringe a mutuos.
    """
    Gd = read_polblogs(path)
    n_raw, l_raw = Gd.number_of_nodes(), Gd.number_of_edges()
    Gu = collapse_to_undirected(Gd)
    Gu = remove_selfloops_and_multi(Gu)
    Gu = graph_gcc_simple(Gu)
    Gu = nx.convert_node_labels_to_integers(Gu)
    out_all = outdir / "edges_all.txt"
    write_edgelist_txt(Gu, out_all)
    st = stats_json(Gu)
    (outdir / "stats.json").write_text(json.dumps({
        "dataset": "Political Blogs (colapsado no dirigido)",
        "n_raw_directed": n_raw, "l_raw_directed": l_raw,
        "n_gcc": st["N"], "l_gcc": st["L"],
        "avg_k": st["avg_k"], "r_kk": st["r_kk"]
    }, indent=2), encoding="utf-8")
    return PrepResult("PoliticalBlogs", n_raw, l_raw, st["N"], st["L"], st["r_kk"], out_all, None, 0.0)

def prep_reactome_local(path: Path, outdir: Path, logfile: Path,
                        source_col: Optional[str], target_col: Optional[str]) -> PrepResult:
    """
    Reactome PPI (local): NO dirigido → limpieza + GCC.
    """
    G = read_reactome_pairs(path, source_col, target_col)
    n_raw, l_raw = G.number_of_nodes(), G.number_of_edges()
    G = remove_selfloops_and_multi(G)
    G = graph_gcc_simple(G)
    G = nx.convert_node_labels_to_integers(G)
    out_all = outdir / "edges_all.txt"
    write_edgelist_txt(G, out_all)
    st = stats_json(G)
    (outdir / "stats.json").write_text(json.dumps({
        "dataset": "Reactome PPI (local)",
        "n_raw": n_raw, "l_raw": l_raw,
        "n_gcc": st["N"], "l_gcc": st["L"],
        "avg_k": st["avg_k"], "r_kk": st["r_kk"]
    }, indent=2), encoding="utf-8")
    return PrepResult("Reactome", n_raw, l_raw, st["N"], st["L"], st["r_kk"], out_all, None, 0.0)

# ================================== CLI ======================================

def parse_args():
    ap = argparse.ArgumentParser(
        description="Descarga y preprocesa datasets reales para Fig. 4 (edge lists)."
    )
    ap.add_argument("--all", action="store_true", help="Procesar TODOS los datasets soportados (descarga+prepro).")
    ap.add_argument("--out", type=str, default="data-prep", help="Carpeta raíz de salida (por dataset).")
    ap.add_argument("--raw", type=str, default="data-raw", help="Carpeta para archivos descargados.")
    ap.add_argument("--enron", action="store_true", help="Descargar+preprocesar Enron (SNAP).")
    ap.add_argument("--hepth", action="store_true", help="Descargar+preprocesar ca-HepTh (SNAP).")
    ap.add_argument("--twitter-higgs", action="store_true", help="Descargar+preprocesar Twitter–Higgs followers (mutuos).")
    ap.add_argument("--polblogs", type=str, default="", help="Ruta local a polblogs.gml o edge list dirigido.")
    ap.add_argument("--reactome", type=str, default="", help="Ruta local a archivo PPI (2 col o CSV/TSV).")
    ap.add_argument("--reactome-src-col", type=str, default="", help="Nombre de columna fuente (si CSV/TSV).")
    ap.add_argument("--reactome-tgt-col", type=str, default="", help="Nombre de columna destino (si CSV/TSV).")
    ap.add_argument("--reactome-download", action="store_true", help="Descargar y preparar Reactome PPI humano (PSI-MITAB).")
    ap.add_argument("--digg", action="store_true", help="Descargar+preprocesar Digg 2009 (amistades; MUTUOS).")
    return ap.parse_args()

def main():
    args = parse_args()
    out_root = Path(args.out)
    raw_root = Path(args.raw)
    out_root.mkdir(parents=True, exist_ok=True)
    raw_root.mkdir(parents=True, exist_ok=True)

    script_name = Path(sys.argv[0]).name
    # Log general de la corrida
    log_all = out_root / "prep_global.log"
    log(log_all, f"{script_name} | inicio | out={out_root} raw={raw_root}")

    # Si no se seleccionó nada, asumimos --all
    selected = any([
        args.enron,
        args.hepth,
        args.twitter_higgs,
        bool(args.polblogs),
        bool(args.reactome),
        args.reactome_download,
        args.digg
    ])
    do_all = args.all or not selected

    if do_all:
        args.enron = True
        args.hepth = True
        args.twitter_higgs = True
        # Para Reactome: usa download a MITAB humano por defecto
        args.reactome_download = True
        # Para Political Blogs: si no entregaron ruta local, activamos descarga
        if not args.polblogs:
            args.polblogs = "__AUTO__"
        args.digg = True

    if args.enron:
        name = "enron"
        outdir = out_root / name
        logfile = outdir / "ejecucion.log"
        t0 = time.perf_counter()
        try:
            log(logfile, f"{script_name} | Enron | preparando...")
            res = prep_enron(raw_root / name, outdir, logfile)
            dt = time.perf_counter() - t0
            log(logfile, f"Listo Enron | N_raw={res.n_raw}, L_raw={res.l_raw}, "
                         f"N_GCC={res.n_gcc}, L_GCC={res.l_gcc}, r_kk={res.rkk:+.3f}, t_total={dt:.2f}s")
        except Exception as e:
            log(logfile, f"ERROR Enron: {e}")

    if args.hepth:
        name = "hepth"
        outdir = out_root / name
        logfile = outdir / "ejecucion.log"
        t0 = time.perf_counter()
        try:
            log(logfile, f"{script_name} | HepTh | preparando...")
            res = prep_hepth(raw_root / name, outdir, logfile)
            dt = time.perf_counter() - t0
            log(logfile, f"Listo HepTh | N_raw={res.n_raw}, L_raw={res.l_raw}, "
                         f"N_GCC={res.n_gcc}, L_GCC={res.l_gcc}, r_kk={res.rkk:+.3f}, t_total={dt:.2f}s")
        except Exception as e:
            log(logfile, f"ERROR HepTh: {e}")

    if args.twitter_higgs:
        name = "twitter-higgs"
        outdir = out_root / name
        logfile = outdir / "ejecucion.log"
        t0 = time.perf_counter()
        try:
            log(logfile, f"{script_name} | Twitter–Higgs | preparando (mutuos followers)...")
            res = prep_twitter_higgs(raw_root / name, outdir, logfile)
            dt = time.perf_counter() - t0
            log(logfile, f"Listo Twitter–Higgs | N_raw_dir={res.n_raw}, L_raw_dir={res.l_raw}, "
                         f"N_GCC={res.n_gcc}, L_GCC={res.l_gcc}, r_kk={res.rkk:+.3f}, t_total={dt:.2f}s")
        except Exception as e:
            log(logfile, f"ERROR Twitter–Higgs: {e}")

    if args.polblogs:
        name = "polblogs"
        outdir = out_root / name
        logfile = outdir / "ejecucion.log"
        t0 = time.perf_counter()
        try:
            if args.polblogs == "__AUTO__":
                log(logfile, f"{script_name} | Political Blogs (download) | preparando ...")
                res = prep_polblogs_download(raw_root / name, outdir, logfile)
            else:
                p = Path(args.polblogs)
                log(logfile, f"{script_name} | Political Blogs (local) | leyendo {p} ...")
                res = prep_polblogs_local(p, outdir, logfile)
            dt = time.perf_counter() - t0
            log(logfile, f"Listo Blogs | N_raw_dir={res.n_raw}, L_raw_dir={res.l_raw}, "
                         f"N_GCC={res.n_gcc}, L_GCC={res.l_gcc}, r_kk={res.rkk:+.3f}, t_total={dt:.2f}s")
        except Exception as e:
            log(logfile, f"ERROR Blogs: {e}")

    if args.reactome:
        name = "reactome"
        outdir = out_root / name
        logfile = outdir / "ejecucion.log"
        t0 = time.perf_counter()
        try:
            p = Path(args.reactome)
            log(logfile, f"{script_name} | Reactome | leyendo {p} ...")
            src_col = args.reactome_src_col or None
            tgt_col = args.reactome_tgt_col or None
            res = prep_reactome_local(p, outdir, logfile, src_col, tgt_col)
            dt = time.perf_counter() - t0
            log(logfile, f"Listo Reactome | N_raw={res.n_raw}, L_raw={res.l_raw}, "
                         f"N_GCC={res.n_gcc}, L_GCC={res.l_gcc}, r_kk={res.rkk:+.3f}, t_total={dt:.2f}s")
        except Exception as e:
            log(logfile, f"ERROR Reactome: {e}")

    if args.reactome_download:
        name = "reactome"
        outdir = out_root / name
        logfile = outdir / "ejecucion.log"
        t0 = time.perf_counter()
        try:
            log(logfile, f"{Path(sys.argv[0]).name} | Reactome (download) | preparando...")
            res = prep_reactome_download(raw_root / name, outdir, logfile)
            dt = time.perf_counter() - t0
            log(logfile, f"Listo Reactome (download) | N_raw={res.n_raw}, L_raw={res.l_raw}, "
                         f"N_GCC={res.n_gcc}, L_GCC={res.l_gcc}, r_kk={res.rkk:+.3f}, t_total={dt:.2f}s")
        except Exception as e:
            log(logfile, f"ERROR Reactome (download): {e}")
    if args.digg:
        name = "digg"
        outdir = out_root / name
        logfile = outdir / "ejecucion.log"
        t0 = time.perf_counter()
        try:
            log(logfile, f"{Path(sys.argv[0]).name} | Digg (download) | preparando...")
            res = prep_digg(raw_root / name, outdir, logfile)
            dt = time.perf_counter() - t0
            log(logfile, f"Listo Digg | N_raw_dir={res.n_raw}, L_raw_dir={res.l_raw}, "
                         f"N_GCC={res.n_gcc}, L_GCC={res.l_gcc}, r_kk={res.rkk:+.3f}, t_total={dt:.2f}s")
        except Exception as e:
            log(logfile, f"ERROR Digg: {e}")

    log(log_all, f"{script_name} | fin")

if __name__ == "__main__":
    main()
