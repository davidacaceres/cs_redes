"""
visualizacion
-------------

Módulo para visualización de redes y resultados de análisis.

Este módulo proporciona funciones para:
- Exportar resultados a diferentes formatos (CSV, JSON, HTML)
- Crear paneles de visualización de métricas
- Generar reportes HTML completos

Ejemplos de uso
---------------

Exportar resumen a CSV::

    from visualizacion import exportar_resultados
    import pandas as pd
    
    resumen_df = pd.DataFrame(...)  # DataFrame con métricas
    exportar_resultados(resumen_df, formato="csv", ruta="resultados.csv")

Generar reporte HTML::

    from visualizacion import generar_reporte_html
    
    generar_reporte_html(
        grafos={"ciudad1": grafo1},
        df_resumen=resumen_df,
        ruta_salida="reporte.html"
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple
import json

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Backend sin GUI para generación de imágenes
import base64

# Importar GrafoSimple para type hints
from preparar_redes import GrafoSimple
import networkx as nx


def generar_grafo_topologico(
    grafo: GrafoSimple,
    ax: plt.Axes,
    max_nodos: int = 100
) -> None:
    """Genera una visualización topológica simplificada del grafo.
    
    Simplificación:
    - Elimina nodos de grado 2 (estaciones intermedias).
    - Mantiene Terminales (grado 1) y Transferencias (grado > 2).
    
    Colores:
    - Terminales: Blanco
    - Transferencias: Naranja
    """
    # Convertir a networkx
    G = nx.Graph()
    G.add_nodes_from(grafo.nodos())
    G.add_edges_from(grafo.aristas())
    
    # Simplificación: Eliminar nodos de grado 2
    # Iteramos hasta que no queden nodos de grado 2 o no se pueda reducir más
    cambios = True
    while cambios:
        cambios = False
        nodos_grado_2 = [n for n in G.nodes() if G.degree(n) == 2]
        
        for n in nodos_grado_2:
            if G.has_node(n): # Verificar si sigue existiendo
                vecinos = list(G.neighbors(n))
                if len(vecinos) == 2:
                    u, v = vecinos
                    # Agregar arista directa entre vecinos
                    if not G.has_edge(u, v):
                        G.add_edge(u, v)
                    # Eliminar nodo intermedio
                    G.remove_node(n)
                    cambios = True

    n_nodos = G.number_of_nodes()
    
    if n_nodos == 0:
        ax.text(0.5, 0.5, "Grafo vacío", ha='center', va='center')
        ax.axis('off')
        return

    # Extraer posiciones geográficas (lon, lat) de los nodos originales
    pos = {}
    nodos_sin_pos = []
    for n in G.nodes():
        attrs = grafo.atributos_nodos.get(n, {})
        if 'lon' in attrs and 'lat' in attrs:
            pos[n] = (attrs['lon'], attrs['lat'])
        else:
            nodos_sin_pos.append(n)
            
    # Si faltan posiciones, usar layout automático para esos nodos (o todo si está vacío)
    if not pos:
        if n_nodos < 50:
            pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
        else:
            pos = nx.kamada_kawai_layout(G) if n_nodos < 200 else nx.spring_layout(G, k=0.2, iterations=20)
    elif nodos_sin_pos:
        # Mezcla de nodos con y sin posición (poco probable en este dataset)
        pos_spring = nx.spring_layout(G, k=0.5, seed=42)
        for n in nodos_sin_pos:
            pos[n] = pos_spring[n]

    # Renumerar nodos secuencialmente (1..N) para visualización
    mapping = {node: i for i, node in enumerate(G.nodes(), 1)}
    G = nx.relabel_nodes(G, mapping)
    
    # Actualizar claves del diccionario pos para coincidir con nuevos IDs
    # Solo si pos tiene claves originales
    if pos:
        pos = {mapping.get(k, k): v for k, v in pos.items() if k in mapping}
        
    # Ajustar aspecto para no distorsionar mapa
    ax.set_aspect('equal')

    # Coloreado según grado en grafo simplificado
    node_colors = []
    for n in G.nodes():
        grado = G.degree(n)
        if grado == 1:
            node_colors.append('white')   # Terminal
        else:
            node_colors.append('#FFC107') # Transferencia / Intersección (Naranja)
    
    # Estilo
    edge_color = 'blue'
    node_size = 500 if n_nodos < 50 else 300
    font_size = 10 if n_nodos < 50 else 8
    
    # Dibujar
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, 
                          node_size=node_size, edgecolors='black', linewidths=1)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_color, width=1.5, alpha=0.7)
    
    if n_nodos < 100:
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=font_size, font_family='sans-serif')
    
    ax.axis('off')
    ax.set_title("Representación Topológica", fontsize=12, fontweight='bold')

def generar_grafico_robustez(
    resultados_robustez: Dict[str, Dict[str, List[float]]],
    ruta_salida: Path
) -> None:
    """Genera y guarda el gráfico de curvas de robustez.
    
    Parámetros
    ----------
    resultados_robustez : Dict
        Diccionario con resultados de robustez (x, y para cada estrategia).
    ruta_salida : Path
        Ruta donde guardar la imagen.
    """
    print(f"[INFO] Generando gráfico de robustez en: {ruta_salida}")
    
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    
    fig = Figure(figsize=(10, 6))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    
    colores = {'grado': 'red', 'aleatorio': 'blue'}
    estilos = {'grado': '-', 'aleatorio': '--'}
    
    for estrategia, datos in resultados_robustez.items():
        if 'x' in datos and 'y' in datos:
            # Normalizar eje Y si es necesario (asumiendo que y es tamaño absoluto)
            # Si queremos fracción, deberíamos dividir por el máximo inicial
            y_vals = datos['y']
            max_val = max(y_vals) if y_vals else 1
            y_norm = [y / max_val for y in y_vals] if max_val > 0 else y_vals
            
            ax.plot(datos['x'], y_norm, 
                   label=f"Ataque {estrategia.capitalize()}", 
                   color=colores.get(estrategia, 'gray'),
                   linestyle=estilos.get(estrategia, '-'),
                   linewidth=2, marker='o', markersize=4)
            
    ax.set_xlabel('Número de Nodos Removidos', fontsize=12)
    ax.set_ylabel('Fracción del Componente Gigante', fontsize=12)
    ax.set_title('Análisis de Robustez de la Red', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    fig.tight_layout()
    fig.savefig(ruta_salida, dpi=150, bbox_inches='tight')
    print(f"[INFO] Gráfico de robustez guardado")


def generar_grafico_distribucion_grados(
    grafo: GrafoSimple,
    ruta_salida: Path
) -> None:
    """Genera y guarda el histograma de distribución de grados.
    
    Parámetros
    ----------
    grafo : GrafoSimple
        El grafo a analizar.
    ruta_salida : Path
        Ruta donde guardar la imagen.
    """
    print(f"[INFO] Generando gráfico de distribución de grados en: {ruta_salida}")
    
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    import numpy as np
    
    grados = list(grafo.grado().values())
    if not grados:
        return

    fig = Figure(figsize=(10, 6))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    
    # Calcular histograma
    counts, bins = np.histogram(grados, bins=range(min(grados), max(grados) + 2))
    
    ax.bar(bins[:-1], counts, width=0.8, color='#17a2b8', edgecolor='black', alpha=0.7)
    
    ax.set_xlabel('Grado (k)', fontsize=12)
    ax.set_ylabel('Frecuencia (N_k)', fontsize=12)
    ax.set_title(f'Distribución de Grados: {grafo.nombre}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Añadir etiquetas de valor sobre las barras
    for i, count in enumerate(counts):
        if count > 0:
            ax.text(bins[i], count + 0.1, str(count), ha='center', va='bottom', fontsize=8)
            
    fig.tight_layout()
    fig.savefig(ruta_salida, dpi=150, bbox_inches='tight')
    print(f"[INFO] Gráfico de distribución de grados guardado")


def exportar_resultados(
    df_resumen: pd.DataFrame,
    formato: str = "csv",
    ruta: Optional[Path] = None
) -> None:
    """Exporta el DataFrame de resumen a diferentes formatos.
    
    Parámetros
    ----------
    df_resumen : pd.DataFrame
        DataFrame con las métricas calculadas.
    formato : str, opcional
        Formato de exportación: "csv", "json", "html" (por defecto "csv").
    ruta : Path, opcional
        Ruta del archivo de salida. Si es None, usa un nombre por defecto.
    
    Ejemplos
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"nombre": ["red1"], "n_nodos": [100]})
    >>> exportar_resultados(df, formato="csv", ruta=Path("resultados.csv"))
    """
    if ruta is None:
        ruta = Path(f"resumen_redes.{formato}")
    
    print(f"[INFO] Exportando resultados a {ruta} (formato: {formato})")
    
    if formato == "csv":
        df_resumen.to_csv(ruta, index=False)
    elif formato == "json":
        df_resumen.to_json(ruta, orient="records", indent=2)
    elif formato == "html":
        df_resumen.to_html(ruta, index=False)
    else:
        raise ValueError(f"Formato no soportado: {formato}. Use 'csv', 'json' o 'html'")
    
    print(f"[INFO] Resultados exportados exitosamente a {ruta}")


def crear_panel_metricas(df_resumen: pd.DataFrame) -> str:
    """Crea un panel de texto con estadísticas resumidas de las métricas.
    
    Parámetros
    ----------
    df_resumen : pd.DataFrame
        DataFrame con las métricas calculadas.
    
    Retorna
    -------
    str
        String con el panel formateado.
    
    Ejemplos
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"nombre": ["red1"], "n_nodos": [100], "n_aristas": [200]})
    >>> panel = crear_panel_metricas(df)
    >>> print(panel)
    """
    panel = []
    panel.append("=" * 80)
    panel.append("PANEL DE MÉTRICAS DE REDES")
    panel.append("=" * 80)
    panel.append(f"\nTotal de redes analizadas: {len(df_resumen)}")
    
    if len(df_resumen) > 0:
        # Estadísticas de nodos
        if "n_nodos" in df_resumen.columns:
            panel.append(f"\nNodos:")
            panel.append(f"  - Promedio: {df_resumen['n_nodos'].mean():.1f}")
            panel.append(f"  - Mínimo: {df_resumen['n_nodos'].min()}")
            panel.append(f"  - Máximo: {df_resumen['n_nodos'].max()}")
        
        # Estadísticas de aristas
        if "n_aristas" in df_resumen.columns:
            panel.append(f"\nAristas:")
            panel.append(f"  - Promedio: {df_resumen['n_aristas'].mean():.1f}")
            panel.append(f"  - Mínimo: {df_resumen['n_aristas'].min()}")
            panel.append(f"  - Máximo: {df_resumen['n_aristas'].max()}")
        
        # Estadísticas de robustez
        if "r_T" in df_resumen.columns:
            panel.append(f"\nIndicador de robustez r_T:")
            panel.append(f"  - Promedio: {df_resumen['r_T'].mean():.4f}")
            panel.append(f"  - Mínimo: {df_resumen['r_T'].min():.4f}")
            panel.append(f"  - Máximo: {df_resumen['r_T'].max():.4f}")
        
        if "C_G" in df_resumen.columns:
            panel.append(f"\nConductancia efectiva C_G:")
            panel.append(f"  - Promedio: {df_resumen['C_G'].mean():.4f}")
            panel.append(f"  - Mínimo: {df_resumen['C_G'].min():.4f}")
            panel.append(f"  - Máximo: {df_resumen['C_G'].max():.4f}")
        
        # Top 5 redes por número de nodos
        if "n_nodos" in df_resumen.columns and "nombre" in df_resumen.columns:
            panel.append(f"\nTop 5 redes por número de nodos:")
            top5 = df_resumen.nlargest(5, "n_nodos")[["nombre", "n_nodos"]]
            for idx, row in top5.iterrows():
                panel.append(f"  {row['nombre']}: {row['n_nodos']} nodos")
    
    panel.append("\n" + "=" * 80)
    
    return "\n".join(panel)


def grafico_comparativo_robustez(df_resumen: pd.DataFrame) -> str:
    """Genera un resumen textual comparativo de robustez.
    
    Parámetros
    ----------
    df_resumen : pd.DataFrame
        DataFrame con las métricas calculadas.
    
    Retorna
    -------
    str
        String con el resumen comparativo.
    
    Ejemplos
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     "nombre": ["red1", "red2"],
    ...     "r_T": [0.5, 0.3],
    ...     "C_G": [0.8, 0.6]
    ... })
    >>> resumen = grafico_comparativo_robustez(df)
    >>> print(resumen)
    """
    resumen = []
    resumen.append("=" * 80)
    resumen.append("COMPARATIVO DE ROBUSTEZ")
    resumen.append("=" * 80)
    
    if len(df_resumen) == 0:
        resumen.append("\nNo hay datos para mostrar")
        return "\n".join(resumen)
    
    # Redes más robustas por r_T
    if "r_T" in df_resumen.columns and "nombre" in df_resumen.columns:
        resumen.append(f"\nRedes más robustas por indicador r_T:")
        top_rT = df_resumen.nlargest(5, "r_T")[["nombre", "r_T"]]
        for idx, row in top_rT.iterrows():
            resumen.append(f"  {row['nombre']}: r_T = {row['r_T']:.4f}")
    
    # Redes más robustas por C_G
    if "C_G" in df_resumen.columns and "nombre" in df_resumen.columns:
        resumen.append(f"\nRedes más robustas por conductancia C_G:")
        top_CG = df_resumen.nlargest(5, "C_G")[["nombre", "C_G"]]
        for idx, row in top_CG.iterrows():
            resumen.append(f"  {row['nombre']}: C_G = {row['C_G']:.4f}")
    
    # Buscar columnas de robustez por remoción
    rob_cols = [col for col in df_resumen.columns if col.startswith("robustez_")]
    if rob_cols and "nombre" in df_resumen.columns:
        for col in rob_cols:
            resumen.append(f"\nRedes más robustas por {col}:")
            top_rob = df_resumen.nlargest(5, col)[["nombre", col]]
            for idx, row in top_rob.iterrows():
                resumen.append(f"  {row['nombre']}: {row[col]:.4f}")
    
    resumen.append("\n" + "=" * 80)
    
    return "\n".join(resumen)


def generar_reporte_html(
    grafos: Dict[str, GrafoSimple],
    df_resumen: pd.DataFrame,
    ruta_salida: Path
) -> None:
    """Genera un reporte HTML completo con todas las métricas y análisis.
    
    Parámetros
    ----------
    grafos : Dict[str, GrafoSimple]
        Diccionario de grafos analizados.
    df_resumen : pd.DataFrame
        DataFrame con las métricas calculadas.
    ruta_salida : Path
        Ruta del archivo HTML de salida.
    
    Ejemplos
    --------
    >>> from pathlib import Path
    >>> generar_reporte_html(
    ...     grafos={"red1": grafo1},
    ...     df_resumen=df,
    ...     ruta_salida=Path("reporte.html")
    ... )
    """
    print(f"[INFO] Generando reporte HTML en {ruta_salida}")
    
    html = []
    html.append("<!DOCTYPE html>")
    html.append("<html lang='es'>")
    html.append("<head>")
    html.append("    <meta charset='UTF-8'>")
    html.append("    <meta name='viewport' content='width=device-width, initial-scale=1.0'>")
    html.append("    <title>Reporte de Análisis de Redes</title>")
    html.append("    <style>")
    html.append("        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }")
    html.append("        h1 { color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }")
    html.append("        h2 { color: #555; margin-top: 30px; }")
    html.append("        table { border-collapse: collapse; width: 100%; margin-top: 20px; background-color: white; }")
    html.append("        th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }")
    html.append("        th { background-color: #4CAF50; color: white; }")
    html.append("        tr:nth-child(even) { background-color: #f2f2f2; }")
    html.append("        .stats { background-color: white; padding: 15px; margin: 20px 0; border-radius: 5px; }")
    html.append("        .stats p { margin: 5px 0; }")
    html.append("    </style>")
    html.append("</head>")
    html.append("<body>")
    html.append("    <h1>Reporte de Análisis de Redes de Transporte Público</h1>")
    
    # Resumen general
    html.append("    <div class='stats'>")
    html.append(f"        <h2>Resumen General</h2>")
    html.append(f"        <p><strong>Total de redes analizadas:</strong> {len(grafos)}</p>")
    html.append(f"        <p><strong>Total de nodos:</strong> {df_resumen['n_nodos'].sum() if 'n_nodos' in df_resumen.columns else 'N/A'}</p>")
    html.append(f"        <p><strong>Total de aristas:</strong> {df_resumen['n_aristas'].sum() if 'n_aristas' in df_resumen.columns else 'N/A'}</p>")
    html.append("    </div>")
    
    # Tabla de métricas
    html.append("    <h2>Métricas Detalladas</h2>")
    html.append(df_resumen.to_html(index=False, classes='table'))
    
    html.append("</body>")
    html.append("</html>")
    
    with open(ruta_salida, "w", encoding="utf-8") as f:
        f.write("\n".join(html))
    
    print(f"[INFO] Reporte HTML generado exitosamente en {ruta_salida}")


def visualizar_red(
    grafo: GrafoSimple,
    disposicion: str = "spring",
    titulo: str = "Red de Transporte"
) -> str:
    """Genera una representación textual de la red.
    
    Nota: Esta es una implementación básica que retorna información textual.
    Para visualización gráfica real, se requeriría matplotlib o similar.
    
    Parámetros
    ----------
    grafo : GrafoSimple
        El grafo a visualizar.
    disposicion : str, opcional
        Tipo de disposición (no implementado en versión textual).
    titulo : str, opcional
        Título de la visualización.
    
    Retorna
    -------
    str
        Representación textual del grafo.
    
    Ejemplos
    --------
    >>> from preparar_redes import GrafoSimple
    >>> G = GrafoSimple()
    >>> G.agregar_arista(1, 2)
    >>> info = visualizar_red(G, titulo="Mi Red")
    >>> print(info)
    """
    info = []
    info.append("=" * 60)
    info.append(f"{titulo}")
    info.append("=" * 60)
    info.append(f"Nodos: {grafo.numero_de_nodos()}")
    info.append(f"Aristas: {grafo.numero_de_aristas()}")
    info.append(f"Conectado: {'Sí' if grafo.esta_conectado() else 'No'}")
    
    if grafo.numero_de_nodos() > 0:
        grados = grafo.grado()
        info.append(f"Grado promedio: {sum(grados.values()) / len(grados):.2f}")
        info.append(f"Grado máximo: {max(grados.values())}")
        info.append(f"Grado mínimo: {min(grados.values())}")
    
    info.append("=" * 60)
    
    return "\n".join(info)


def generar_mapa_geografico(
    grafo: GrafoSimple,
    ruta_salida: Optional[Path] = None,
    mostrar_nombres: bool = False
) -> Optional[plt.Figure]:
    """Genera un mapa geográfico de la red con estaciones y líneas.
    
    Parámetros
    ----------
    grafo : GrafoSimple
        El grafo a visualizar.
    ruta_salida : Path, opcional
        Si se especifica, guarda la imagen en esta ruta.
    mostrar_nombres : bool, opcional
        Si True, muestra nombres de estaciones (puede saturar el mapa).
    
    Retorna
    -------
    Optional[plt.Figure]
        Figura de matplotlib, o None si no hay datos geográficos.
    
    Ejemplos
    --------
    >>> from preparar_redes import GrafoSimple
    >>> from pathlib import Path
    >>> G = GrafoSimple()
    >>> # ... agregar nodos con lat/lon ...
    >>> fig = generar_mapa_geografico(G, Path("mapa.png"))
    """
    print(f"[INFO] Generando mapa geográfico para: {grafo.nombre}")
    
    # Extraer coordenadas
    lats = []
    lons = []
    nodos_con_coords = []
    
    for nodo in grafo.nodos():
        attrs = grafo.atributos_nodos.get(nodo, {})
        if 'lat' in attrs and 'lon' in attrs:
            lats.append(attrs['lat'])
            lons.append(attrs['lon'])
            nodos_con_coords.append(nodo)
    
    if not lats:
        print(f"[ADVERTENCIA] No hay datos geográficos para {grafo.nombre}")
        return None
    
    print(f"[INFO] Encontradas {len(lats)} estaciones con coordenadas")
    
    # Crear figura usando API orientada a objetos (Thread-Safe)
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    
    fig = Figure(figsize=(12, 10))
    FigureCanvasAgg(fig) # Adjuntar canvas Agg
    ax = fig.add_subplot(111)
    
    # Plotear líneas primero (para que queden detrás)
    print(f"[INFO] Dibujando conexiones...")
    from matplotlib.collections import LineCollection
    segmentos = []
    lineas_dibujadas = 0
    for u, v in grafo.aristas():
        u_attrs = grafo.atributos_nodos.get(u, {})
        v_attrs = grafo.atributos_nodos.get(v, {})
        if 'lat' in u_attrs and 'lat' in v_attrs and 'lon' in u_attrs and 'lon' in v_attrs:
            segmentos.append([
                (u_attrs['lon'], u_attrs['lat']),
                (v_attrs['lon'], v_attrs['lat'])
            ])
            lineas_dibujadas += 1
            
    if segmentos:
        lc = LineCollection(segmentos, colors='blue', alpha=0.3, linewidths=0.8, zorder=1)
        ax.add_collection(lc)
    
    print(f"[INFO] Dibujadas {lineas_dibujadas} conexiones")
    
    # Plotear estaciones
    ax.scatter(lons, lats, c='red', s=60, alpha=0.7, zorder=2, edgecolors='darkred', linewidths=0.5)
    
    # Opcional: mostrar nombres
    if mostrar_nombres and len(nodos_con_coords) < 100:  # Solo si no son demasiados
        for nodo in nodos_con_coords:
            attrs = grafo.atributos_nodos.get(nodo, {})
            if 'name' in attrs:
                ax.annotate(
                    attrs['name'],
                    (attrs['lon'], attrs['lat']),
                    fontsize=6,
                    alpha=0.6
                )
    
    ax.set_xlabel('Longitud', fontsize=12)
    ax.set_ylabel('Latitud', fontsize=12)
    ax.set_title(f'Mapa de Red: {grafo.nombre}\n{len(lats)} estaciones, {lineas_dibujadas} conexiones', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Ajustar límites con margen
    if lats and lons:
        lat_margin = (max(lats) - min(lats)) * 0.1 or 0.01
        lon_margin = (max(lons) - min(lons)) * 0.1 or 0.01
        ax.set_xlim(min(lons) - lon_margin, max(lons) + lon_margin)
        ax.set_ylim(min(lats) - lat_margin, max(lats) + lat_margin)
    
    # Agregar mapa base de Google Satellite
    try:
        import contextily as cx
        print("[INFO] Agregando mapa base Google Satellite...")
        # crs=4326 indica que nuestros datos (puntos/líneas) están en lat/lon (WGS84)
        # contextily transformará automáticamente los tiles para coincidir
        
        # Google Maps Satellite URL
        google_sat_url = "https://mt0.google.com/vt/lyrs=s&hl=en&x={x}&y={y}&z={z}&s=Ga"
        
        cx.add_basemap(
            ax, 
            crs=4326, 
            source=google_sat_url,
            attribution="Google Maps"
        )
    except Exception as e:
        print(f"[ADVERTENCIA] No se pudo agregar mapa base: {e}")

    fig.tight_layout()
    
    if ruta_salida:
        print(f"[INFO] Guardando mapa en: {ruta_salida}")
        fig.savefig(ruta_salida, dpi=150, bbox_inches='tight')
        print(f"[INFO] Mapa guardado exitosamente")
    
    return fig


def crear_directorio_salida(
    nombre_red: str,
    directorio_base: Path = Path("procesados")
) -> Path:
    """Crea la estructura de directorios para una red.
    
    Crea:
    - procesados/{nombre_red}/
    - procesados/{nombre_red}/imagenes/
    - procesados/{nombre_red}/datos/
    - procesados/{nombre_red}/reportes/
    
    Parámetros
    ----------
    nombre_red : str
        Nombre de la red.
    directorio_base : Path, opcional
        Directorio base (por defecto "procesados").
    
    Retorna
    -------
    Path
        Ruta al directorio de la red.
    
    Ejemplos
    --------
    >>> from pathlib import Path
    >>> dir_salida = crear_directorio_salida("adelaide", Path("data/procesados"))
    >>> print(dir_salida)
    data/procesados/adelaide
    """
    dir_red = directorio_base / nombre_red
    dir_imagenes = dir_red / "imagenes"
    dir_datos = dir_red / "datos"
    dir_reportes = dir_red / "reportes"
    
    # Crear directorios
    dir_imagenes.mkdir(parents=True, exist_ok=True)
    dir_datos.mkdir(parents=True, exist_ok=True)
    dir_reportes.mkdir(parents=True, exist_ok=True)
    
    print(f"[INFO] Estructura de directorios creada en: {dir_red}")
    
    return dir_red


def guardar_resultados_red(
    grafo: GrafoSimple,
    metricas: Dict,
    directorio_salida: Path,
    generar_mapa: bool = True,
    nombre_html: str = "reporte.html",
    grafo_metricas: Optional[GrafoSimple] = None
) -> Dict[str, Path]:
    """Guarda todos los resultados de análisis de una red.
    metricas : Dict
        Diccionario con las métricas calculadas.
    directorio_salida : Path
        Directorio donde guardar los resultados.
    generar_mapa : bool, opcional
        Si True, genera el mapa geográfico (por defecto True).
    nombre_html : str, opcional
        Nombre del archivo HTML (por defecto "reporte.html").
    grafo_metricas : GrafoSimple, opcional
        Grafo a usar para gráficos estadísticos (robustez, grados).
        Si es None, se usa 'grafo'.
    
    Retorna
    -------
    Dict[str, Path]
        Diccionario con las rutas de los archivos generados.
    
    Ejemplos
    --------
    >>> from preparar_redes import GrafoSimple
    >>> from pathlib import Path
    >>> G = GrafoSimple()
    >>> metricas = {"n_nodos": 100, "n_aristas": 200}
    >>> dir_salida = Path("procesados/adelaide")
    >>> archivos = guardar_resultados_red(G, metricas, dir_salida)
    """
    if grafo_metricas is None:
        grafo_metricas = grafo
    print(f"[INFO] Guardando resultados para: {grafo.nombre}")
    archivos_generados = {}
    
    # 1. Guardar mapa geográfico
    if generar_mapa:
        ruta_mapa = directorio_salida / "imagenes" / "mapa.png"
        fig = generar_mapa_geografico(grafo, ruta_mapa)
        if fig is not None:
            archivos_generados['mapa'] = ruta_mapa
            
    # 1.1 Guardar mapa topológico
    ruta_topologia = directorio_salida / "imagenes" / "topologia.png"
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig_topo = Figure(figsize=(10, 10))
    FigureCanvasAgg(fig_topo)
    ax_topo = fig_topo.add_subplot(111)
    generar_grafo_topologico(grafo, ax_topo)
    fig_topo.savefig(ruta_topologia, dpi=150, bbox_inches='tight')
    archivos_generados['topologia'] = ruta_topologia
    print(f"[INFO] Mapa topológico guardado en: {ruta_topologia}")

    # 1.2 Guardar gráfico de robustez (si las métricas lo incluyen)
    # Calculamos la curva de robustez explícitamente para asegurar que el gráfico se genere
    from procesar_redes import calcular_curva_robustez
    print(f"[INFO] Calculando curva de robustez para gráfico...")
    datos_robustez = calcular_curva_robustez(grafo_metricas)
    ruta_robustez = directorio_salida / "imagenes" / "robustez.png"
    generar_grafico_robustez(datos_robustez, ruta_robustez)
    archivos_generados['robustez'] = ruta_robustez
    
    # 1.3 Guardar gráfico de distribución de grados
    ruta_grados = directorio_salida / "imagenes" / "distribucion_grados.png"
    generar_grafico_distribucion_grados(grafo_metricas, ruta_grados)
    archivos_generados['distribucion_grados'] = ruta_grados
    
    # 2. Guardar métricas en JSON
    ruta_json = directorio_salida / "datos" / "metricas.json"
    with open(ruta_json, 'w', encoding='utf-8') as f:
        json.dump(metricas, f, indent=2, ensure_ascii=False, default=str)
    archivos_generados['metricas_json'] = ruta_json
    print(f"[INFO] Métricas guardadas en JSON: {ruta_json}")
    
    # 3. Guardar métricas en CSV
    ruta_csv = directorio_salida / "datos" / "metricas.csv"
    df_metricas = pd.DataFrame([metricas])
    df_metricas.to_csv(ruta_csv, index=False)
    archivos_generados['metricas_csv'] = ruta_csv
    print(f"[INFO] Métricas guardadas en CSV: {ruta_csv}")
    
    # 4. Generar reporte HTML
    # Si el nombre es index.html, lo guardamos en la raíz del directorio_salida, no en reportes/
    if nombre_html == "index.html":
        ruta_html = directorio_salida / nombre_html
    else:
        ruta_html = directorio_salida / "reportes" / nombre_html
        
    _generar_reporte_individual(grafo, metricas, ruta_html)
    archivos_generados['reporte_html'] = ruta_html
    
    print(f"[INFO] Resultados guardados exitosamente en: {directorio_salida}")
    return archivos_generados


def imagen_a_base64(ruta_imagen: Path) -> str:
    """Convierte una imagen a string base64."""
    if not ruta_imagen.exists():
        print(f"[DEBUG] Imagen NO encontrada: {ruta_imagen.absolute()}")
        return ""
    try:
        with open(ruta_imagen, "rb") as img_file:
            print(f"[DEBUG] Codificando imagen: {ruta_imagen.absolute()}")
            return base64.b64encode(img_file.read()).decode("utf-8")
    except Exception as e:
        print(f"[WARN] No se pudo convertir imagen a base64 ({ruta_imagen}): {e}")
        return ""





def _generar_reporte_individual(
    grafo: GrafoSimple,
    metricas: Dict,
    ruta_salida: Path,
    directorio_origen_imagenes: Optional[Path] = None,
    ruta_img_radar: Optional[Path] = None,
    ruta_img_robustez: Optional[Path] = None
) -> None:
    """Genera un reporte HTML para una red individual."""
    
    # Formatear métricas a 2 decimales si son float
    metricas_fmt = {}
    for k, v in metricas.items():
        # Excluir curva_robustez del reporte tabular
        if k == 'curva_robustez':
            continue
            
        if isinstance(v, float):
            metricas_fmt[k] = f"{v:.2f}"
        else:
            metricas_fmt[k] = v
            
    # Intentar cargar info de la ciudad (Markdown)
    # Asumimos estructura: sitio/analisis/<red>/index.html -> ../../info/<red>.md
    # O si estamos en procesados: procesados/<red>/reportes/reporte.html -> ??? (No aplica para el sitio)
    
    # Detectar si estamos en modo "sitio"
    es_sitio = "analisis" in str(ruta_salida)
    
    info_content = ""
    ruta_flag = ""
    
    if es_sitio:
        # Rutas relativas para el sitio
        ruta_info_md = ruta_salida.parent.parent.parent / "info" / f"{grafo.nombre}.md"
        ruta_flag_img = ruta_salida.parent.parent.parent / "banderas" / f"{grafo.nombre}.png"
        
        if ruta_info_md.exists():
            try:
                with open(ruta_info_md, "r", encoding="utf-8") as f:
                    info_content = f.read()
                # Convertir saltos de línea a <br> o párrafos simples
                info_content = info_content.replace("\n### ", "<h3>").replace("\n", "<br>")
                # Cerrar h3 si es necesario (muy básico)
                info_content = info_content.replace("<h3>", "<h3>").replace("<br><h3>", "</h3><h3>")
            except Exception as e:
                print(f"[WARN] No se pudo leer info para {grafo.nombre}: {e}")
        
        if ruta_flag_img.exists():
            ruta_flag = ruta_flag_img
            
    # Intentar buscar también usando directorio_origen_imagenes si existe (para modo single report)
    # Suponemos estructura: data/procesados/M_red -> data/info/M_red.md
    if not info_content and directorio_origen_imagenes:
        # directorio_origen_imagenes apunta a .../procesados/<red>
        # info estaría en .../procesados/../../info
        try:
             # Subir 2 niveles desde procesados/<red> -> data
             dir_data = directorio_origen_imagenes.parent.parent
             # Asumimos que 'sitio' es hermano de 'data' (estructura del proyecto)
             dir_sitio = dir_data.parent / "sitio"
             
             ruta_info = dir_sitio / "info" / f"{grafo.nombre}.md"
             ruta_flg = dir_sitio / "banderas" / f"{grafo.nombre}.png"
             
             # Fallback: si no está en sitio, buscar en data/info (por si acaso)
             if not ruta_info.exists():
                 ruta_info = dir_data / "info" / f"{grafo.nombre}.md"
             if not ruta_flg.exists():
                 ruta_flg = dir_data / "banderas" / f"{grafo.nombre}.png"
             
             if ruta_info.exists():
                 with open(ruta_info, "r", encoding="utf-8") as f:
                    info_content = f.read()
                    info_content = info_content.replace("\n### ", "<h3>").replace("\n", "<br>")
                    info_content = info_content.replace("<h3>", "<h3>").replace("<br><h3>", "</h3><h3>")
             
             if ruta_flg.exists():
                 ruta_flag = ruta_flg
             else:
                # Búsqueda Fuzzy de Bandera
                try:
                    nombre_limpio = grafo.nombre.lower().replace("kujala_", "").replace("metro_", "").replace("m_", "").replace("_l", "")
                    
                    # Buscar en carpetas de banderas (sitio y data)
                    dirs_buscar = [dir_sitio / "banderas", dir_data / "banderas"]
                    for d in dirs_buscar:
                        if d.exists():
                            candidatos = list(d.glob(f"*{nombre_limpio}.png"))
                            if candidatos:
                                mejor = candidatos[0]
                                for c in candidatos:
                                    if "-" in c.name:
                                        mejor = c
                                        break
                                ruta_flag = mejor
                                break
                except Exception as ex:
                    print(f"[WARN] Error en búsqueda fuzzy de bandera: {ex}")

        except Exception as e:
            print(f"[WARN] Error buscando info extra: {e}")
            ruta_flag = ruta_flag_img
    
    # Preparar imágenes en Base64
    dir_base = ruta_salida.parent
    if dir_base.name == "reportes":
        dir_base = dir_base.parent
        
    # Si se especificó un directorio origen (ej: para exportación manual), usarlo
    if directorio_origen_imagenes:
        dir_base = directorio_origen_imagenes
        print(f"[DEBUG] Usando directorio de imagenes explícito: {dir_base}")
    
    ruta_mapa = dir_base / "imagenes" / "mapa.png"
    ruta_topo = dir_base / "imagenes" / "topologia.png"
    
    print(f"[DEBUG] Buscando mapa en: {ruta_mapa.absolute()}")
    print(f"[DEBUG] Buscando topologia en: {ruta_topo.absolute()}")
    
    b64_flag = imagen_a_base64(ruta_flag) if isinstance(ruta_flag, Path) else ""
    b64_mapa = imagen_a_base64(ruta_mapa)
    b64_topo = imagen_a_base64(ruta_topo)
    
    b64_radar = imagen_a_base64(ruta_img_radar) if ruta_img_radar else ""
    b64_robustez = imagen_a_base64(ruta_img_robustez) if ruta_img_robustez else ""
            
    html = []
    html.append("<!DOCTYPE html>")
    html.append("<html lang='es'>")
    html.append("<head>")
    html.append("    <meta charset='UTF-8'>")
    html.append("    <meta name='viewport' content='width=device-width, initial-scale=1.0'>")
    html.append(f"    <title>Reporte de Red: {grafo.nombre}</title>")
    html.append("    <style>")
    html.append("        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 0; background-color: #f4f7f6; color: #333; }")
    html.append("        header { background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); color: white; padding: 2rem; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }")
    html.append("        h1 { margin: 0; font-size: 2.5rem; font-weight: 300; }")
    html.append("        .container { max-width: 1200px; margin: 2rem auto; padding: 0 1rem; }")
    html.append("        .card { background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); padding: 2rem; margin-bottom: 2rem; }")
    html.append("        h2 { color: #1e3c72; border-bottom: 2px solid #e0e0e0; padding-bottom: 0.5rem; margin-top: 0; }")
    html.append("        .metrics-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 1.5rem; }")
    html.append("        .metric-item { background: #f8f9fa; padding: 1rem; border-radius: 6px; border-left: 4px solid #2a5298; }")
    html.append("        .metric-label { display: block; font-size: 0.85rem; color: #666; margin-bottom: 0.25rem; }")
    html.append("        .metric-value { display: block; font-size: 1.25rem; font-weight: bold; color: #333; }")
    html.append("        .img-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 2rem; }")
    html.append("        .img-container { background: white; padding: 1rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); text-align: center; }")
    html.append("        img { max-width: 100%; height: auto; border-radius: 4px; }")
    html.append("        .flag-header { display: flex; align-items: center; justify-content: center; gap: 1rem; margin-bottom: 1rem; }")
    html.append("        .flag-img { height: 60px; width: auto; border-radius: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.2); }")
    html.append("        .info-section { line-height: 1.6; }")
    html.append("        .info-section { line-height: 1.6; }")
    html.append("        .info-section h3 { color: #2a5298; margin-top: 1.5rem; }")
    html.append("        ")
    html.append("        /* New Layout Styles */")
    html.append("        .intro-grid { display: grid; grid-template-columns: 1fr 2fr; gap: 2rem; margin-bottom: 2rem; }")
    html.append("        @media (max-width: 768px) { .intro-grid { grid-template-columns: 1fr; } }")
    html.append("        .intro-flag-container { background: white; padding: 1rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); display: flex; align-items: center; justify-content: center; }")
    html.append("        .intro-flag-img { max-height: 200px; width: auto; max-width: 100%; border-radius: 4px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }")
    html.append("    </style>")
    html.append("</head>")
    html.append("<body>")
    
    html.append("    <header>")
    html.append("        <div class='flag-header'>")
    # Flag movido al body, dejamos solo header limpio o pequeño logo si se desea
    # html.append(f"            <img src='data:image/png;base64,{b64_flag}' alt='Bandera' class='flag-img'>")
    html.append(f"            <h1>Análisis de Red: {grafo.nombre}</h1>")
    html.append("        </div>")
    html.append("        <p style='margin-top: 0.5rem; opacity: 0.9;'>Reporte generado automáticamente</p>")
    html.append("    </header>")
    
    # Sección de Intro (Bandera + Info)
    html.append("    <div class='container'>")
    html.append("    <div class='intro-grid'>")
    
    # Columna 1: Bandera
    html.append("        <div class='intro-flag-container'>") 
    html.append("            <div style='text-align:center; width:100%;'>")
    html.append("                <h3 style='margin-top:0;'>Bandera / Escudo</h3>")
    if b64_flag:
        html.append(f"                <img src='data:image/png;base64,{b64_flag}' alt='Bandera' class='intro-flag-img'>")
    else:
        html.append("                <p>No disponible</p>")
    html.append("            </div>")
    html.append("        </div>")
    
    # Columna 2: Información
    html.append("        <div class='card info-section' style='margin-bottom:0;'>")
    html.append("            <h2 style='margin-top:0;'>Información de la Ciudad</h2>")
    if info_content:
        html.append(f"            <div>{info_content}</div>")
    else:
        html.append("            <p>Información no disponible.</p>")
    html.append("        </div>")
    
    html.append("    </div>")
    
    # Métricas (Tabla General)
    html.append("        <div class='card'>")
    html.append("            <h2>Información General y Métricas de Robustez</h2>")
    
    # Lista ordenada de métricas (Siglas solicitadas)
    # Metros (Nombre) ya se muestra en título, pero lo incluimos si se desea en tabla
    keys_ordenadas = [
        "Metros", "N", "L", "r̄ᵀ", "C_G", "Rel_G", "E[1/H]", "CC_G", "μ̄ₙ₋₁", 
        "Ē[D]", "λ̄*", "1/κ", "M_G", "f₉₀%-degree", "f₉₀%-random", 
        "f_c-degree", "f_c-random", "Area"
    ]
    
    html.append("            <div class='metrics-grid' style='grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));'>")
    
    for k in keys_ordenadas:
        if k in metricas_fmt:
            val = metricas_fmt[k]
        elif k in metricas: # Buscar en raw si no está fmt
             if isinstance(metricas[k], float):
                 val = f"{metricas[k]:.3f}" # 3 decimales
             else:
                 val = str(metricas[k])
        else:
            continue # O saltar si no existe
            
        html.append("                <div class='metric-item'>")
        # Usar sigla tal cual
        html.append(f"                    <span class='metric-label'>{k}</span>")
        html.append(f"                    <span class='metric-value' style='font-size: 1.1rem;'>{val}</span>")
        html.append("                </div>")
        
    html.append("            </div>")
    html.append("        </div>")
    
    # Imágenes (Mapa y Topología)
    html.append("        <div class='card'>")
    html.append("            <h2>Visualización</h2>")
    html.append("            <div class='img-grid'>")
    
    # Radar
    html.append("                <div class='img-container'>")
    html.append("                    <h3 style='margin-top:0'>Gráfico Radar</h3>")
    if b64_radar:
         html.append(f"                    <img src='data:image/png;base64,{b64_radar}' alt='Radar Chart'>")
    else:
         html.append("                    <p>No disponible</p>")
    html.append("                </div>")
    
    # Robustez
    html.append("                <div class='img-container'>")
    html.append("                    <h3 style='margin-top:0'>Análisis Robustez</h3>")
    if b64_robustez:
         html.append(f"                    <img src='data:image/png;base64,{b64_robustez}' alt='Robustness Chart'>")
    else:
         html.append("                    <p>No disponible</p>")
    html.append("                </div>")

    # Mapa
    html.append("                <div class='img-container'>")
    html.append("                    <h3 style='margin-top:0'>Mapa Geográfico</h3>")
    if b64_mapa:
         html.append(f"                    <img src='data:image/png;base64,{b64_mapa}' alt='Mapa Geográfico'>")
    else:
         html.append("                    <p>No disponible</p>")
    html.append("                </div>")
    
    # Topología
    html.append("                <div class='img-container'>")
    html.append("                    <h3 style='margin-top:0'>Topología (Simplificada k!=2)</h3>")
    if b64_topo:
         html.append(f"                    <img src='data:image/png;base64,{b64_topo}' alt='Grafo Topológico'>")
    else:
         html.append("                    <p>No disponible</p>")
    html.append("                </div>")
    
    html.append("            </div>")
    html.append("        </div>")
    
    html.append("    </div>")
    

    
    with open(ruta_salida, "w", encoding="utf-8") as f:
        f.write("\n".join(html))
    print(f"[INFO] Reporte individual generado: {ruta_salida}")
    
    # Abrir automáticamente en el navegador
    try:
        import webbrowser
        webbrowser.open(ruta_salida.as_uri())
        print(f"[INFO] Abriendo reporte en navegador...")
    except Exception as e:
        print(f"[WARN] No se pudo abrir el navegador automáticamente: {e}")


def generar_tabla_resumen_html(df: pd.DataFrame, ruta_salida: Path):
    """Genera una tabla HTML comparativa con todas las redes y métricas.
    Incluye tooltips con las descripciones solicitadas.
    """
    import pandas as pd # Ensure pandas is available locally if strictly needed, though usually top-level
    
    # Definición de columnas y descripciones (Tooltips)
    cols_info = {
        "Metros": "La ciudad a la que pertenece la red de metro analizada.",
        "N": "Número de nodos (estaciones).",
        "L": "Número de enlaces (vías férreas).",
        "r̄ᵀ": "Indicador de robustez normalizado r̄ᵀ. Cuantifica la robustez en términos de caminos alternativos por nodo.",
        "C_G": "Conductancia efectiva C_G. Mide robustez basándose en redundancia y longitud de caminos.",
        "Rel_G": "Fiabilidad Rel_G. Probabilidad de que la red esté conectada ante fallos aleatorios de enlaces.",
        "E[1/H]": "Eficiencia promedio E[1/H]. Cuantifica la eficiencia de transporte global (inverso de distancias).",
        "CC_G": "Coeficiente de agrupamiento CC_G. Mide la densidad de conexión entre los vecinos de un nodo.",
        "μ̄ₙ₋₁": "Conectividad algebraica normalizada μ̄ₙ₋₁. Segundo valor propio del Laplaciano (dificultad de desconexión).",
        "Ē[D]": "Grado promedio normalizado Ē[D]. Conectividad promedio relativa al tamaño de la red.",
        "λ̄*": "Conectividad natural normalizada λ̄*. Caracteriza redundancia de rutas y robustez estructural.",
        "1/κ": "Inversa de diversidad de grado 1/κ. Relacionada con el umbral de percolación (resiliencia a desintegración).",
        "M_G": "Coeficiente de mallado M_G. Mide la estructura de ciclos en grafos planares.",
        "f₉₀%-degree": "Umbral f₉₀% (Ataque Grado). Fracción de nodos a eliminar (por grado) para reducir LCC al 90%.",
        "f₉₀%-random": "Umbral f₉₀% (Fallo Aleatorio). Fracción de nodos a eliminar (aleatorio) para reducir LCC al 90%.",
        "f_c-degree": "Umbral f_c (Ataque Grado). Fracción de nodos a eliminar para desintegrar la red.",
        "f_c-random": "Umbral f_c (Fallo Aleatorio). Fracción de nodos a eliminar para desintegrar la red.",
        "Area": "Área del polígono de robustez. Evaluación general combinando las 10 métricas teóricas."
    }
    
    # Orden deseado
    orden_cols = list(cols_info.keys())
    
    # Estructura HTML
    html = [
        "<!DOCTYPE html>",
        "<html lang='es'>",
        "<head>",
        "    <meta charset='UTF-8'>",
        "    <title>Tabla Resumen de Robustez</title>",
        "    <style>",
        "        body { font-family: 'Segoe UI', Arial, sans-serif; padding: 20px; background: #f5f5f5; color: #333; }",
        "        h1 { text-align: center; color: #333; margin-bottom: 20px; }",
        "        .table-wrapper { overflow-x: auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }",
        "        table { width: 100%; border-collapse: collapse; font-size: 0.9rem; min-width: 1200px; }",
        "        th, td { padding: 12px 10px; text-align: center; border-bottom: 1px solid #e0e0e0; }",
        "        th { background-color: #f8f9fa; font-weight: 600; position: sticky; top: 0; z-index: 10; border-bottom: 2px solid #007bff; color: #007bff; cursor: help; }",
        "        th:hover { background-color: #e9ecef; }",
        "        tr:nth-child(even) { background-color: #f8f9fa; }",
        "        tr:hover { background-color: #f1f4f8; transition: background-color 0.2s; }",
        "        .tooltip-icon { font-size: 0.8em; color: #999; margin-left: 4px; }",
        "    </style>",
        "</head>",
        "<body>",
        "    <h1>Tabla 1: Métricas de Robustez</h1>",
        "    <div class='table-wrapper'>",
        "        <table>",
        "            <thead>",
        "                <tr>",
        "                    <th>N°</th>"
    ]
    
    # Headers
    for col in orden_cols:
        desc = cols_info.get(col, "")
        html.append(f"                    <th title='{desc}'>{col}</th>")
    
    html.append("                </tr>")
    html.append("            </thead>")
    html.append("            <tbody>")
    
    # Rows
    # Ordenar por nombre ciudad si es posible
    if "Metros" in df.columns:
        df = df.sort_values("Metros")
    elif "nombre" in df.columns:
        df = df.sort_values("nombre")
        
    count = 1
    for _, row in df.iterrows():
        html.append("                <tr>")
        html.append(f"                    <td>{count}</td>")
        
        for col in orden_cols:
            val = row.get(col, "-")
            # Formato
            if isinstance(val, (float, int)) and not isinstance(val, bool):
                 if isinstance(val, float):
                     val_str = f"{val:.3f}"
                 else:
                     val_str = str(val)
            else:
                 val_str = str(val)
            
            desc = cols_info.get(col, "")
            # Tooltip en celda
            html.append(f"                    <td title='{col}\\n{desc}'>{val_str}</td>")
            
        html.append("                </tr>")
        count += 1
        
    html.append("            </tbody>")
    html.append("        </table>")
    html.append("    </div>")
    html.append("    <p style='text-align:center; color:#666; margin-top:20px;'>Deslice el cursor sobre los encabezados o celdas para ver el significado de cada métrica.</p>")
    html.append("</body>")
    html.append("</html>")
    
    try:
        with open(ruta_salida, "w", encoding="utf-8") as f:
            f.write("\n".join(html))
        print(f"[INFO] Tabla resumen guardada en: {ruta_salida}")
    except Exception as e:
        print(f"[ERROR] No se pudo guardar tabla resumen: {e}")

