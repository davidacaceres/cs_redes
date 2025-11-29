#!/usr/bin/env python3
"""
tarea-3
=======

Script principal para análisis de redes de transporte público.

Este script orquesta el flujo completo de trabajo:
1. Descarga de datasets (Kujala y Metro51)
2. Extracción y organización de datos
3. Construcción de grafos con procesamiento paralelo
4. Cálculo de métricas y análisis de robustez
5. Visualización y exportación de resultados

El script utiliza los siguientes módulos especializados:
- preparar_datos: Descarga y extracción de datasets
- preparar_redes: Construcción de grafos con threading
- procesar_redes: Cálculo de métricas con threading
- visualizacion: Exportación y visualización de resultados

Uso
---
Ejecutar con configuración por defecto::

    python tarea-3.py

Especificar directorio de datos::

    python tarea-3.py --data-dir mi_carpeta_datos

Incluir dataset de 51 metros::

    python tarea-3.py --include-metro51

Exportar resultados a CSV::

    python tarea-3.py --output-csv resultados.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

# Importar módulos especializados
import preparar_datos
import preparar_redes
import procesar_redes
import visualizacion

# Importar interfaz gráfica (solo si se solicita)
try:
    import interfaz_grafica
    _HAS_GUI = True
except ImportError:
    _HAS_GUI = False


def main(argv: list[str] | None = None) -> None:
    """Punto de entrada principal del script.

    Este comando permite descargar los datos, extraerlos, cargar
    redes y calcular métricas. El uso típico consiste en ejecutar
    ``python tarea-3.py`` para procesar los datasets disponibles.
    """
    parser = argparse.ArgumentParser(
        description="Ejecuta análisis de redes de transporte público (Tarea 3)."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Carpeta base para guardar los datos descargados."
    )
    parser.add_argument(
        "--download",
        action="store_true",
        default=False,
        help="Descargar datasets desde Internet."
    )
    parser.add_argument(
        "--compute",
        action="store_true",
        default=True,
        help="Calcular métricas y generar resumen."
    )
    parser.add_argument(
        "--include-metro51",
        action="store_true",
        default=False,
        help="Incluir el dataset de 51 metros si está disponible."
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Ruta para escribir el resumen en CSV."
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        default=None,
        help="Ruta para generar reporte HTML."
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        default=False,
        help="Desactivar procesamiento paralelo."
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        default=True,
        help="Ejecutar en modo interfaz gráfica (GUI)."
    )
    
    args = parser.parse_args(argv)

    # Modo GUI
    if args.gui:
        if not _HAS_GUI:
            print("[ERROR] No se pudo importar el módulo interfaz_grafica")
            print("Asegúrese de que tkinter y PIL están instalados")
            return
        
        print("[INFO] Iniciando interfaz gráfica...")
        app = interfaz_grafica.VentanaPrincipal(directorio_datos=args.data_dir)
        app.ejecutar()
        return
    
    # Modo consola (código existente)

    # Configurar rutas
    base_dir = args.data_dir
    kujala_zip_dir = base_dir / "kujala_zips"
    kujala_extract_dir = base_dir / "kujala"
    metro_zip_path = base_dir / "metro51.zip"
    metro_extract_dir = base_dir / "51_metro_networks"

    # Fase 1: Descarga de datos
    if args.download:
        print("\n" + "=" * 80)
        print("FASE 1: DESCARGA DE DATOS")
        print("=" * 80)
        
        # Descargar 25 ciudades de Kujala
        preparar_datos.descargar_ciudades_kujala(kujala_zip_dir)
        
        # Extraer datasets
        preparar_datos.extraer_datasets(
            kujala_zip_dir,
            kujala_extract_dir,
            metro_zip_path,
            metro_extract_dir
        )

    # Fase 2: Construcción de redes (con procesamiento paralelo)
    if args.compute:
        print("\n" + "=" * 80)
        print("FASE 2: CONSTRUCCIÓN DE REDES")
        print("=" * 80)
        
        usar_paralelo = not args.no_parallel
        
        # Cargar redes de Kujala
        grafos_kujala, nodos_kujala = preparar_redes.cargar_dataset_kujala(
            raiz=base_dir / "kujala" / "procesado",
            usar_paralelo=usar_paralelo
        )
        
        # Cargar redes de Metro51 si se solicita
        grafos_metro = {}
        nodos_metro = {}
        if args.include_metro51 and metro_extract_dir.exists():
            grafos_metro, nodos_metro = preparar_redes.cargar_metros_desde_carpeta(
                raiz=metro_extract_dir,
                usar_paralelo=usar_paralelo
            )
        
        # Combinar todos los grafos con prefijos
        todos_grafos: Dict[str, preparar_redes.GrafoSimple] = {}
        for nombre, G in grafos_kujala.items():
            todos_grafos[f"K_{nombre}"] = G
        for nombre, G in grafos_metro.items():
            todos_grafos[f"M_{nombre}"] = G
        
        if not todos_grafos:
            print("\n[ERROR] No se encontraron redes. Asegúrate de descargar los datos o revisar las rutas.")
            return
        
        print(f"\n[INFO] Total de redes cargadas: {len(todos_grafos)}")
        print(f"  - Kujala: {len(grafos_kujala)} redes")
        print(f"  - Metro51: {len(grafos_metro)} redes")
        
        # Fase 3: Procesamiento de métricas (con procesamiento paralelo)
        print("\n" + "=" * 80)
        print("FASE 3: CÁLCULO DE MÉTRICAS")
        print("=" * 80)
        
        resumen = procesar_redes.calcular_resumen_dataset(
            grafos=todos_grafos,
            fraccion_remover=0.2,
            ejecuciones_aleatorias=10,
            semilla=42,
            usar_paralelo=usar_paralelo
        )
        
        # Fase 4: Visualización y exportación
        print("\n" + "=" * 80)
        print("FASE 4: VISUALIZACIÓN Y EXPORTACIÓN")
        print("=" * 80)
        
        # Mostrar panel de métricas
        panel = visualizacion.crear_panel_metricas(resumen)
        print("\n" + panel)
        
        # Mostrar comparativo de robustez
        comparativo = visualizacion.grafico_comparativo_robustez(resumen)
        print("\n" + comparativo)
        
        # Exportar a CSV si se solicita
        if args.output_csv is not None:
            visualizacion.exportar_resultados(
                resumen,
                formato="csv",
                ruta=args.output_csv
            )
        
        # Generar reporte HTML si se solicita
        if args.output_html is not None:
            visualizacion.generar_reporte_html(
                grafos=todos_grafos,
                df_resumen=resumen,
                ruta_salida=args.output_html
            )
        
        # Mostrar resumen en pantalla
        print("\n" + "=" * 80)
        print("RESUMEN DE MÉTRICAS (primeras 10 redes)")
        print("=" * 80)
        print(resumen.head(10).to_string(index=False))
        
        print("\n" + "=" * 80)
        print("ANÁLISIS COMPLETADO")
        print("=" * 80)
        print(f"Total de redes analizadas: {len(resumen)}")
        if args.output_csv:
            print(f"Resultados guardados en: {args.output_csv}")
        if args.output_html:
            print(f"Reporte HTML generado en: {args.output_html}")


if __name__ == "__main__":
    main()