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

# Importar GrafoSimple para type hints
from preparar_redes import GrafoSimple


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
    generar_mapa: bool = True
) -> Dict[str, Path]:
    """Guarda todos los resultados de análisis de una red.
    
    Guarda:
    - Mapa geográfico (si hay datos): imagenes/mapa.png
    - Métricas en JSON: datos/metricas.json
    - Métricas en CSV: datos/metricas.csv
    - Reporte HTML: reportes/reporte.html
    
    Parámetros
    ----------
    grafo : GrafoSimple
        El grafo analizado.
    metricas : Dict
        Diccionario con las métricas calculadas.
    directorio_salida : Path
        Directorio donde guardar los resultados.
    generar_mapa : bool, opcional
        Si True, genera el mapa geográfico (por defecto True).
    
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
    print(f"[INFO] Guardando resultados para: {grafo.nombre}")
    archivos_generados = {}
    
    # 1. Guardar mapa geográfico
    if generar_mapa:
        ruta_mapa = directorio_salida / "imagenes" / "mapa.png"
        fig = generar_mapa_geografico(grafo, ruta_mapa)
        if fig is not None:
            archivos_generados['mapa'] = ruta_mapa
            # plt.close(fig) no es necesario con API orientada a objetos
    
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
    ruta_html = directorio_salida / "reportes" / "reporte.html"
    _generar_reporte_individual(grafo, metricas, ruta_html)
    archivos_generados['reporte_html'] = ruta_html
    
    print(f"[INFO] Resultados guardados exitosamente en: {directorio_salida}")
    return archivos_generados


def _generar_reporte_individual(
    grafo: GrafoSimple,
    metricas: Dict,
    ruta_salida: Path
) -> None:
    """Genera un reporte HTML para una red individual."""
    html = []
    html.append("<!DOCTYPE html>")
    html.append("<html lang='es'>")
    html.append("<head>")
    html.append("    <meta charset='UTF-8'>")
    html.append("    <meta name='viewport' content='width=device-width, initial-scale=1.0'>")
    html.append(f"    <title>Reporte: {grafo.nombre}</title>")
    html.append("    <style>")
    html.append("        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }")
    html.append("        h1 { color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }")
    html.append("        h2 { color: #555; margin-top: 30px; }")
    html.append("        .metric { background-color: white; padding: 15px; margin: 10px 0; border-radius: 5px; }")
    html.append("        .metric strong { color: #4CAF50; }")
    html.append("        img { max-width: 100%; height: auto; border: 1px solid #ddd; margin-top: 20px; }")
    html.append("    </style>")
    html.append("</head>")
    html.append("<body>")
    html.append(f"    <h1>Reporte de Análisis: {grafo.nombre}</h1>")
    
    # Mapa si existe
    ruta_mapa = ruta_salida.parent.parent / "imagenes" / "mapa.png"
    if ruta_mapa.exists():
        html.append("    <h2>Mapa de la Red</h2>")
        html.append(f"    <img src='../imagenes/mapa.png' alt='Mapa de la red'>")
    
    # Métricas
    html.append("    <h2>Métricas Calculadas</h2>")
    for clave, valor in metricas.items():
        html.append(f"    <div class='metric'><strong>{clave}:</strong> {valor}</div>")
    
    html.append("</body>")
    html.append("</html>")
    
    with open(ruta_salida, "w", encoding="utf-8") as f:
        f.write("\n".join(html))
    
    print(f"[INFO] Reporte HTML individual generado: {ruta_salida}")
