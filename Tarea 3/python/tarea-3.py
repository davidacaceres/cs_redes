#!/usr/bin/env python3
"""
tarea-3
=======

Script principal para análisis de redes de transporte público.
Este script inicia la interfaz gráfica para el análisis.

Uso
---
    python tarea-3.py
"""

from __future__ import annotations
from pathlib import Path
import sys

# Importar interfaz gráfica
try:
    import interfaz_grafica
    _HAS_GUI = True
except ImportError as e:
    _HAS_GUI = False
    _IMPORT_ERROR = str(e)

def main(argv: list[str] | None = None) -> None:
    """Punto de entrada principal del script."""
    
    if not _HAS_GUI:
        print("[ERROR] No se pudo importar el módulo interfaz_grafica")
        print(f"Detalle: {_IMPORT_ERROR}")
        print("Asegúrese de que tkinter, PIL, matplotlib y pandas están instalados")
        return
    
    # Directorio base de datos hardcoded a "data"
    data_dir = Path("data")
    
    if not data_dir.exists():
        print(f"[ADVERTENCIA] El directorio de datos '{data_dir}' no existe.")
        print("Creando directorio vacío...")
        data_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Iniciando interfaz gráfica...")
    app = interfaz_grafica.VentanaPrincipal(directorio_datos=data_dir)
    app.ejecutar()

if __name__ == "__main__":
    main()