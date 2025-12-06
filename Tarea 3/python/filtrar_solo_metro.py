"""
filtrar_solo_metro.py
---------------------

Script para filtrar el dataset Kujala dejando solo estaciones de metro/tren.

Crea archivos CSV limpios en data/kujala/clean/ con únicamente:
- Nodos: Estaciones de tren (Railway Station)
- Aristas: Conexiones de tipo metro/tren (route_type=2)
"""

import os
import pandas as pd
from pathlib import Path

def filtrar_ciudad(ciudad, dir_origen, dir_destino):
    """
    Filtra los datos de una ciudad para quedarse solo con estaciones de metro.
    
    Parámetros
    ----------
    ciudad : str
        Nombre de la ciudad
    dir_origen : Path
        Directorio con datos originales
    dir_destino : Path
        Directorio donde guardar datos filtrados
    
    Retorna
    -------
    dict
        Estadísticas del filtrado
    """
    print(f"\n[INFO] Procesando {ciudad}...")
    
    # Rutas de archivos
    ciudad_origen = dir_origen / ciudad
    ciudad_destino = dir_destino / ciudad
    
    archivo_nodos = ciudad_origen / "network_nodes.csv"
    archivo_aristas = ciudad_origen / "network_combined.csv"
    
    # Verificar que existan los archivos
    if not archivo_nodos.exists() or not archivo_aristas.exists():
        print(f"[WARNING] Archivos no encontrados para {ciudad}, saltando...")
        return None
    
    # Leer archivos
    try:
        nodos = pd.read_csv(archivo_nodos, sep=';')
        aristas = pd.read_csv(archivo_aristas, sep=';')
    except Exception as e:
        print(f"[ERROR] Error leyendo archivos de {ciudad}: {e}")
        return None
    
    # Estadísticas originales
    n_nodos_orig = len(nodos)
    n_aristas_orig = len(aristas)
    
    # FILTRAR NODOS: Solo estaciones de tren (Railway Station)
    nodos_metro = nodos[nodos['name'].str.contains('Railway Station', case=False, na=False)].copy()
    
    # Obtener IDs de nodos válidos
    ids_validos = set(nodos_metro['stop_I'].values)
    
    # FILTRAR ARISTAS: Solo route_type=2 (metro/tren) y entre nodos válidos
    aristas_metro = aristas[
        (aristas['route_type'] == 2) &
        (aristas['from_stop_I'].isin(ids_validos)) &
        (aristas['to_stop_I'].isin(ids_validos))
    ].copy()
    
    # Estadísticas filtradas
    n_nodos_metro = len(nodos_metro)
    n_aristas_metro = len(aristas_metro)
    
    # Crear directorio de destino
    ciudad_destino.mkdir(parents=True, exist_ok=True)
    
    # Guardar archivos filtrados
    nodos_metro.to_csv(ciudad_destino / "network_nodes.csv", sep=';', index=False)
    aristas_metro.to_csv(ciudad_destino / "network_combined.csv", sep=';', index=False)
    
    print(f"  Nodos: {n_nodos_orig} → {n_nodos_metro} ({n_nodos_metro/n_nodos_orig*100:.1f}%)")
    print(f"  Aristas: {n_aristas_orig} → {n_aristas_metro} ({n_aristas_metro/n_aristas_orig*100:.1f}%)")
    
    return {
        'ciudad': ciudad,
        'nodos_orig': n_nodos_orig,
        'nodos_metro': n_nodos_metro,
        'aristas_orig': n_aristas_orig,
        'aristas_metro': n_aristas_metro
    }


def main():
    """Procesa todas las ciudades del dataset Kujala."""
    
    # Directorios
    dir_base = Path("data/kujala")
    dir_origen = dir_base / "procesado"
    dir_destino = dir_base / "clean"
    
    print("="*60)
    print("FILTRADO DE DATASET KUJALA - SOLO ESTACIONES DE METRO")
    print("="*60)
    
    # Obtener lista de ciudades
    ciudades = [d.name for d in dir_origen.iterdir() if d.is_dir()]
    ciudades.sort()
    
    print(f"\n[INFO] Encontradas {len(ciudades)} ciudades")
    print(f"[INFO] Directorio origen: {dir_origen}")
    print(f"[INFO] Directorio destino: {dir_destino}")
    
    # Procesar cada ciudad
    estadisticas = []
    for ciudad in ciudades:
        stats = filtrar_ciudad(ciudad, dir_origen, dir_destino)
        if stats:
            estadisticas.append(stats)
    
    # Resumen final
    print("\n" + "="*60)
    print("RESUMEN FINAL")
    print("="*60)
    
    if estadisticas:
        df_stats = pd.DataFrame(estadisticas)
        
        print(f"\nCiudades procesadas: {len(df_stats)}")
        print(f"\nEstadísticas agregadas:")
        print(f"  Total nodos originales: {df_stats['nodos_orig'].sum():,}")
        print(f"  Total nodos metro: {df_stats['nodos_metro'].sum():,}")
        print(f"  Total aristas originales: {df_stats['aristas_orig'].sum():,}")
        print(f"  Total aristas metro: {df_stats['aristas_metro'].sum():,}")
        
        # Guardar reporte
        archivo_reporte = dir_destino / "reporte_filtrado.csv"
        df_stats.to_csv(archivo_reporte, index=False)
        print(f"\n[INFO] Reporte guardado en: {archivo_reporte}")
        
        # Mostrar top 10 ciudades por número de estaciones
        print(f"\nTop 10 ciudades por número de estaciones de metro:")
        top10 = df_stats.nlargest(10, 'nodos_metro')[['ciudad', 'nodos_metro', 'aristas_metro']]
        print(top10.to_string(index=False))
    
    print("\n[INFO] Proceso completado!")


if __name__ == "__main__":
    main()
