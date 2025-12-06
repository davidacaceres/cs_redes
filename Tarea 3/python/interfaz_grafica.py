"""
interfaz_grafica
----------------

Módulo para interfaz gráfica de análisis de redes de transporte público.

Este módulo proporciona una interfaz gráfica completa usando tkinter que permite:
- Seleccionar dataset (Kujala o Metro51)
- Seleccionar ciudad/red específica
- Visualizar mapa geográfico de la red
- Ver métricas y análisis en tabs
- Exportar resultados

La interfaz usa threading para no bloquear la UI durante el procesamiento.

Ejemplos de uso
---------------

Ejecutar la interfaz gráfica::

    from interfaz_grafica import VentanaPrincipal
    
    app = VentanaPrincipal()
    app.ejecutar()
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from pathlib import Path
from typing import Dict, Optional, List
import threading
import queue
import time
import os

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import pandas as pd
import numpy as np

# Importar módulos del proyecto
import preparar_redes
import procesar_redes
import visualizacion
from hmi.mapa import MapaWidget
from hmi.grafo_interactivo import GrafoInteractivo
from PIL import ImageGrab

class VentanaPrincipal:
    """Ventana principal de la aplicación GUI.
    
    Proporciona una interfaz completa para análisis de redes con:
    - Selectores de dataset y ciudad
    - Panel de mapa geográfico
    - Tabs con métricas y análisis
    - Barra de progreso
    """
    
    def __init__(self, directorio_datos: Path = Path("data")):
        """Inicializa la ventana principal.
        
        Parámetros
        ----------
        directorio_datos : Path, opcional
            Directorio base de datos (por defecto "data").
        """
        self.directorio_datos = directorio_datos
        self.directorio_procesados = directorio_datos / "procesados"
        
        # Estado de la aplicación
        self.grafo_actual: Optional[preparar_redes.GrafoSimple] = None
        self.metricas_actuales: Optional[Dict] = None
        self.dir_salida_actual: Optional[Path] = None
        self.figura_mapa: Optional[plt.Figure] = None
        
        # Cola para comunicación con threads
        self.cola_resultados = queue.Queue()
        
        # Flag para cancelar procesamiento
        self.cancelar_procesamiento = False
        self.thread_actual = None
        
        # Variables para tiempo transcurrido
        self.tiempo_inicio_analisis = 0.0
        self.mensaje_estado_actual = "Listo"
        self.evento_stop_timer = threading.Event()
        
        # Crear ventana
        self.ventana = tk.Tk()
        self.ventana.title("Análisis de Redes de Transporte Público")
        self.ventana.geometry("1400x900")
        
        # Crear interfaz
        self.crear_interfaz()
        
        # Iniciar verificación de cola
        self.ventana.after(100, self.verificar_cola)
    
    def crear_interfaz(self):
        """Crea todos los componentes de la interfaz."""
        # Frame principal
        frame_principal = ttk.Frame(self.ventana, padding="10")
        frame_principal.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configurar pesos para redimensionamiento
        self.ventana.columnconfigure(0, weight=1)
        self.ventana.rowconfigure(0, weight=1)
        frame_principal.columnconfigure(1, weight=1)
        frame_principal.rowconfigure(1, weight=1)
        
        # 1. Crear controles superiores
        self.crear_controles(frame_principal)
        
        # 2. Crear paneles (mapa y tabs)
        self.crear_paneles(frame_principal)
        
        # 3. Crear barra de progreso
        self.crear_barra_progreso(frame_principal)
    
    def crear_controles(self, parent):
        """Crea los controles de selección."""
        frame_controles = ttk.LabelFrame(parent, text="Configuración", padding="10")
        frame_controles.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Dataset
        ttk.Label(frame_controles, text="Dataset:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.combo_dataset = ttk.Combobox(frame_controles, values=["Kujala", "Metro51"], state="readonly", width=15)
        self.combo_dataset.grid(row=0, column=1, sticky=tk.W, padx=5)
        self.combo_dataset.current(0)
        self.combo_dataset.bind("<<ComboboxSelected>>", self.on_dataset_changed)
        
        # Ciudad/Red
        ttk.Label(frame_controles, text="Ciudad/Red:").grid(row=0, column=2, sticky=tk.W, padx=5)
        self.combo_ciudad = ttk.Combobox(frame_controles, state="readonly", width=30)
        self.combo_ciudad.grid(row=0, column=3, sticky=tk.W, padx=5)
        
        # Botón Generar
        self.boton_generar = ttk.Button(frame_controles, text="Generar Análisis", command=self.generar_analisis)
        self.boton_generar.grid(row=0, column=4, padx=20)
        
        # Cargar lista de ciudades inicial
        self.cargar_lista_ciudades()
    
    def crear_paneles(self, parent):
        """Crea los paneles de mapa y tabs con divisor redimensionable."""
        # Crear PanedWindow para permitir redimensionamiento
        paned = ttk.PanedWindow(parent, orient=tk.HORIZONTAL)
        paned.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Panel izquierdo: Mapa
        frame_mapa = ttk.LabelFrame(paned, text="Mapa Geográfico", padding="10")
        frame_mapa.columnconfigure(0, weight=1)

        # frame_mapa.rowconfigure(0, weight=1) # Removed to prevent controls from expanding
        
        # Canvas para matplotlib
        # Canvas para matplotlib
        
        # 1. Canvas del mapa (primero para poder pasar self.mapa_widget a los controles)
        self.frame_canvas_mapa = ttk.Frame(frame_mapa)
        self.frame_canvas_mapa.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        frame_mapa.rowconfigure(1, weight=1) # Expand map, not controls
        
        # Label inicial
        self.label_mapa = ttk.Label(self.frame_canvas_mapa, text="Seleccione una red y presione 'Generar Análisis'", 
                                     font=("Arial", 12))
        self.label_mapa.pack(expand=True)
        self.mapa_widget = MapaWidget(self.frame_canvas_mapa)

        # 2. Frame controles mapa
        frame_controles_mapa = ttk.Frame(frame_mapa)
        frame_controles_mapa.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        ttk.Label(frame_controles_mapa, text="Proveedor de Mapa:").pack(side=tk.LEFT, padx=5)
        
        self.combo_proveedor = ttk.Combobox(frame_controles_mapa, state="readonly", width=30)
        self.combo_proveedor["values"] = [
            "OpenStreetMap (Default)",
            "Google Maps Normal",
            "Google Maps Satélite",
            "Google Maps Híbrido",
            "OpenTopoMap"
        ]
        self.combo_proveedor.current(0)
        self.combo_proveedor.pack(side=tk.LEFT, padx=5)
        self.combo_proveedor.bind("<<ComboboxSelected>>", self.cambiar_proveedor_mapa)
        
        # Separador
        ttk.Separator(frame_controles_mapa, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=5, fill=tk.Y)
        
        # Botones de control (vinculados directamente a self.mapa_widget)
        # self.btn_zoom_in = ttk.Button(frame_controles_mapa, text="+", width=3, command=self.mapa_widget.zoom_in)
        # self.btn_zoom_in.pack(side=tk.LEFT, padx=2)
        
        # self.btn_zoom_out = ttk.Button(frame_controles_mapa, text="-", width=3, command=self.mapa_widget.zoom_out)
        # self.btn_zoom_out.pack(side=tk.LEFT, padx=2)
        
        # self.btn_pan = ttk.Button(frame_controles_mapa, text="Pan", width=5, command=self.mapa_widget.activar_pan)
        # self.btn_pan.pack(side=tk.LEFT, padx=2)
        
        self.btn_area = ttk.Button(frame_controles_mapa, text="⛶ Área", width=8, command=self.mapa_widget.activar_zoom_area)
        self.btn_area.pack(side=tk.LEFT, padx=2)
        
        self.btn_centrar = ttk.Button(frame_controles_mapa, text="🎯 Centrar", width=10, command=self.mapa_widget.centrar)
        self.btn_centrar.pack(side=tk.LEFT, padx=2)
        
        # Callback de clic en nodo del mapa
        self.mapa_widget.set_callback_clic_nodo(self.on_clic_nodo_mapa)
        
        # Variables para zoom area (ya no se usan aquí, están en MapaWidget)
        # self.zoom_rect_id = None
        # self.start_x = None
        # self.start_y = None
        # self.original_bounds = None

        # Panel derecho: Tabs
        frame_tabs = ttk.LabelFrame(paned, text="Análisis", padding="10")
        frame_tabs.columnconfigure(0, weight=1)
        frame_tabs.rowconfigure(0, weight=1)
        
        # Notebook (tabs)
        self.notebook = ttk.Notebook(frame_tabs)
        self.notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Agregar paneles al PanedWindow
        paned.add(frame_mapa, weight=1)
        paned.add(frame_tabs, weight=1)
        
        # Tab 1: Información General
        self.crear_tab_info()
        
        # Tab 2: Métricas de Robustez
        self.crear_tab_robustez()
        
        # Tab 3: Componentes
        self.crear_tab_componentes()
        
        # Tab 3.5: Resumen (Tabla)
        self.crear_tab_resumen()

        # Tab 4: Exportar
        self.crear_tab_exportar()
    
    def crear_tab_info(self):
        """Crea el tab de información general con tabla y grafo."""
        frame_info = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(frame_info, text="Info General")
        
        # Configurar grid para división 50/50
        frame_info.columnconfigure(0, weight=1)
        frame_info.rowconfigure(0, weight=1, uniform="split")  # 50%
        frame_info.rowconfigure(1, weight=1, uniform="split")  # 50%
        
        # 1. Frame Superior: Notebook con Tabla y Gráfico Radar
        self.notebook_info_top = ttk.Notebook(frame_info)
        self.notebook_info_top.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # Tab 1.1: Tabla de Métricas
        frame_tabla = ttk.Frame(self.notebook_info_top, padding="5")
        self.notebook_info_top.add(frame_tabla, text="Métricas")
        
        columns = ("Métrica", "Valor")
        self.tree_info = ttk.Treeview(frame_tabla, columns=columns, show="headings", height=8) # Height reducido
        self.tree_info.heading("Métrica", text="Métrica")
        self.tree_info.heading("Valor", text="Valor")
        self.tree_info.column("Métrica", width=250)
        self.tree_info.column("Valor", width=200)
        
        scrollbar = ttk.Scrollbar(frame_tabla, orient=tk.VERTICAL, command=self.tree_info.yview)
        self.tree_info.configure(yscroll=scrollbar.set)
        
        self.tree_info.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Tooltips para tree_info
        self.tree_info.bind("<Motion>", self._on_info_tree_motion)
        self.tree_info.bind("<Leave>", self._hide_tooltip)
        
        # Tab 1.2: Gráfico Radar (Como en la foto)
        self.frame_radar = ttk.Frame(self.notebook_info_top, padding="5")
        self.notebook_info_top.add(self.frame_radar, text="Gráfico Radial")
        
        # Canvas placeholder
        self.canvas_radar_widget = None
        
        # 2. Frame Inferior: Visualización de grafo topológico interactivo
        frame_grafo = ttk.LabelFrame(frame_info, text="Topología de Red", padding="5")
        frame_grafo.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.grafo_interactivo = GrafoInteractivo(frame_grafo)
        self.grafo_interactivo.pack(fill=tk.BOTH, expand=True)
        self.grafo_interactivo.set_callback_doble_clic(self.centrar_mapa_en_nodo)

    def centrar_mapa_en_nodo(self, lat, lon):
        """Centra el mapa en las coordenadas dadas si no están visibles."""
        if hasattr(self, 'mapa_widget') and self.mapa_widget.map_widget:
            map_widget = self.mapa_widget.map_widget
            
            # Verificar si el punto está visible en el viewport actual
            visible = False
            try:
                # Obtener coordenadas de las esquinas del viewport
                top_left = map_widget.convert_canvas_coords_to_decimal_coords(0, 0)
                bottom_right = map_widget.convert_canvas_coords_to_decimal_coords(map_widget.width, map_widget.height)
                
                # top_left es (lat_max, lon_min), bottom_right es (lat_min, lon_max)
                lat_max, lon_min = top_left
                lat_min, lon_max = bottom_right
                
                # Verificar si lat, lon está dentro
                if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
                    visible = True
            except Exception:
                # Si falla el cálculo, asumir no visible
                pass
            
            if not visible:
                map_widget.set_position(lat, lon)
                # Opcional: Hacer zoom in al centrar si está muy lejos? No, mantener zoom usuario.
            
            # Mostrar efecto sonar siempre
            self.mapa_widget.mostrar_sonar(lat, lon)

    def on_clic_nodo_mapa(self, nodo_id):
        """Maneja el clic en un nodo del mapa geográfico."""
        if self.grafo_interactivo:
            self.grafo_interactivo.mostrar_sonar(nodo_id)
            print(f"[INFO] Clic en mapa nodo {nodo_id}. Mostrando sonar en topología.")
    
    def crear_tab_robustez(self):
        """Crea el tab de métricas de robustez."""
        frame_robustez = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(frame_robustez, text="Robustez")
        
        # Frame para canvas de matplotlib
        self.frame_canvas_robustez = ttk.Frame(frame_robustez)
        self.frame_canvas_robustez.pack(fill=tk.BOTH, expand=True)
        
        frame_robustez.columnconfigure(0, weight=1)
        frame_robustez.rowconfigure(0, weight=1)
    
    def crear_tab_componentes(self):
        """Crea el tab de análisis de componentes."""
        frame_componentes = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(frame_componentes, text="Componentes")
        
        # Frame para canvas de matplotlib
        self.frame_canvas_componentes = ttk.Frame(frame_componentes)
        self.frame_canvas_componentes.pack(fill=tk.BOTH, expand=True)
        
        frame_componentes.columnconfigure(0, weight=1)
        frame_componentes.rowconfigure(0, weight=1)

    def crear_tab_resumen(self):
        """Crea el tab de resumen (Tabla General)."""
        frame_resumen = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(frame_resumen, text="Resumen")
        
        # Botón cargar
        # ttk.Button(frame_resumen, text="Recargar Tabla", command=self.cargar_tabla_resumen).pack(pady=(0, 10))
        
        # Treeview Container
        frame_tabla = ttk.Frame(frame_resumen)
        frame_tabla.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbars
        scroll_y = ttk.Scrollbar(frame_tabla, orient=tk.VERTICAL)
        scroll_x = ttk.Scrollbar(frame_tabla, orient=tk.HORIZONTAL)
        
        # Treeview
        self.tree_resumen = ttk.Treeview(
            frame_tabla, 
            yscrollcommand=scroll_y.set, 
            xscrollcommand=scroll_x.set,
            selectmode="browse"
        )
        
        scroll_y.config(command=self.tree_resumen.yview)
        scroll_x.config(command=self.tree_resumen.xview)
        
        scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.tree_resumen.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Tooltip setup
        self.tooltip_label = None
        self.tooltip_window = None
        self.tree_resumen.bind("<Motion>", self._on_tree_motion)
        self.tree_resumen.bind("<Leave>", self._hide_tooltip)
        
        # Descriptions (Hardcoded for simplicity and speed)
        # Descriptions with Interpretations
        self.cols_desc = {
            "Country": "País de la red.",
            "City": "Ciudad de la red.",
            "N": "Número de nodos (estaciones).\nAlto: Red grande.\nBajo: Red pequeña.",
            "L": "Número de enlaces (vías férreas).\nAlto: Red con muchas conexiones.",
            "r_T": "Indicador de robustez normalizado r̄ᵀ.\nAlto: Alta robustez (muchas rutas alternativas).\nBajo: Baja robustez (estructura de árbol).",
            "C_G": "Conductancia efectiva C_G.\nAlto: Alta conectividad y redundancia (difícil de embotellar).\nBajo: Pobre conectividad.",
            "Efficiency": "Eficiencia global promedio E[1/H].\nAlto: Transporte eficiente (pocos saltos entre nodos).\nBajo: Red ineficiente o lineal.",
            "Clustering": "Coeficiente de agrupamiento CC_G.\nAlto: Red con muchos triángulos (comunidades locales).\nBajo: Pocas conexiones entre vecinos.",
            "Alg_Conn": "Conectividad algebraica normalizada μ̄ₙ₋₁.\nAlto: Muy difícil de desconectar (robusta).\nBajo: Fácil de partir en componentes (frágil).",
            "Avg_Deg": "Grado promedio normalizado Ē[D].\nAlto: Red densa.\nBajo: Red dispersa.",
            "Meshedness": "Coeficiente de mallado M_G.\nAlto: Estructura tipo malla con muchos ciclos.\nBajo: Estructura tipo árbol o lineal.",
            "Rel_G": "Fiabilidad (Reliability).\nAlto (prox 1.0): Red confiable ante fallos aleatorios.\nBajo: Red propensa a desconectarse.",
            "Nat_Conn": "Conectividad natural normalizada λ̄*.\nAlto: Alta redundancia de rutas y estabilidad.\nBajo: Baja redundancia.",
            "Kappa": "Diversidad de grado κ.\nAlto: Red heterogénea (hub-and-spoke).\nBajo: Red homogénea (grados similares).",
            "Inv_Kappa": "Inversa de diversidad 1/κ.\nAlto: Red homogénea (percolación difícil).\nBajo: Red heterogénea (hubs vulnerables a ataques).",
            "f90_Targeted": "Umbral ataque grado (90% LCC).\nAlto: Se requiere eliminar muchos hubs para dañar la red (Robusta).\nBajo: Frágil ante ataques.",
            "f90_Random": "Umbral fallo aleatorio (90% LCC).\nAlto: Soporta muchos fallos antes de degradarse.\nBajo: Se degrada rápido.",
            "fc_Targeted": "Umbral colapso (Ataque Grado).\nAlto: Difícil de desintegrar completamente.\nBajo: Fácil de destruir atacando hubs.",
            "fc_Random": "Umbral colapso (Fallo Aleatorio).\nAlto: Muy resiliente a fallos al azar.\nBajo: Se desintegra con pocos fallos.",
            "Area": "Área del polígono de robustez.\nAlto: Red globalmente robusta en todas las métricas.\nBajo: Red frágil en general."
        }
        
        # Aliases para compatibilidad con Tab Análisis (que usa símbolos científicos)
        # y Tab Resumen (que usa claves CSV internas)
        self.cols_desc["Metros"] = self.cols_desc["City"]
        self.cols_desc["r̄ᵀ"] = self.cols_desc["r_T"]
        self.cols_desc["μ̄ₙ₋₁"] = self.cols_desc["Alg_Conn"]
        self.cols_desc["λ̄*"] = self.cols_desc["Nat_Conn"]
        self.cols_desc["E[1/H]"] = self.cols_desc["Efficiency"]
        self.cols_desc["CC_G"] = self.cols_desc["Clustering"]
        self.cols_desc["Ē[D]"] = self.cols_desc["Avg_Deg"]
        self.cols_desc["M_G"] = self.cols_desc["Meshedness"]
        self.cols_desc["1/κ"] = self.cols_desc["Inv_Kappa"]
        self.cols_desc["f₉₀%-degree"] = self.cols_desc["f90_Targeted"]
        self.cols_desc["f₉₀%-random"] = self.cols_desc["f90_Random"]
        self.cols_desc["f_c-degree"] = self.cols_desc["fc_Targeted"]
        self.cols_desc["f_c-random"] = self.cols_desc["fc_Random"]

    def _on_info_tree_motion(self, event):
        """Muestra tooltip en el tree_info (Tab Análisis)."""
        # Identificar item bajo el mouse
        item_id = self.tree_info.identify_row(event.y)
        if not item_id:
            self._hide_tooltip()
            return

        # Obtener valores del item (Métrica, Valor)
        vals = self.tree_info.item(item_id, "values")
        if not vals:
            self._hide_tooltip()
            return
            
        metrica_key = vals[0] # Primera columna es el nombre de la métrica
        
        if hasattr(self, 'cols_desc'):
            desc = self.cols_desc.get(metrica_key, "")
            if desc:
                self._show_tooltip(event.x_root, event.y_root, f"{metrica_key}\n{desc}")
            else:
                self._hide_tooltip()
        else:
            self._hide_tooltip()
        
        # Cargar datos automágicamente
        self.ventana.after(100, self.cargar_tabla_resumen)

    def cargar_tabla_resumen(self):
        """Carga el CSV de resumen en el Treeview."""
        # Ruta hardcoded según solicitud: utils/resultados_metro_robustez_completo.csv
        # Asumiendo estructura de proyecto: python/interfaz_grafica.py -> python/utils/...
        # self.directorio_datos apunta a 'data'
        # El script está en 'python'. utils está en 'python/utils'.
        
        try:
            ruta_csv = Path(__file__).parent / "utils" / "resultados_metro_robustez_completo.csv"
            if not ruta_csv.exists():
                # Fallback: intentar en CWD
                ruta_csv = Path("utils") / "resultados_metro_robustez_completo.csv"
            
            if not ruta_csv.exists():
                print(f"[WARN] No se encontró CSV resumen en {ruta_csv}")
                return

            df = pd.read_csv(ruta_csv)
            
            # Limpiar tree
            self.tree_resumen.delete(*self.tree_resumen.get_children())
            
            # Configurar columnas
            cols = list(df.columns)
            self.tree_resumen["columns"] = cols
            self.tree_resumen["show"] = "headings" # Ocultar columna fantasma #0
            
            # Mapa de nombres CSV a Símbolos Científicos
            header_map = {
                "r_T": "r̄ᵀ",
                "Alg_Conn": "μ̄ₙ₋₁",
                "Nat_Conn": "λ̄*",
                "Efficiency": "E[1/H]",
                "Clustering": "CC_G",
                "Avg_Deg": "Ē[D]",
                "Meshedness": "M_G",
                "Inv_Kappa": "1/κ",
                "f90_Targeted": "f₉₀%-degree",
                "f90_Random": "f₉₀%-random",
                "fc_Targeted": "f_c-degree",
                "fc_Random": "f_c-random"
            }
            
            for col in cols:
                display_name = header_map.get(col, col)
                self.tree_resumen.heading(col, text=display_name)
                # Ajustar ancho (básico)
                width = 80 if len(col) < 5 else 120
                if col in ["Country", "City"]: width = 100
                self.tree_resumen.column(col, width=width, minwidth=50, stretch=False)
            
            # Insertar datos
            for index, row in df.iterrows():
                vals = []
                for v in row:
                    if isinstance(v, float):
                        vals.append(f"{v:.3f}")
                    else:
                        vals.append(str(v))
                self.tree_resumen.insert("", "end", values=vals)
                
        except Exception as e:
            print(f"Error cargando tabla: {e}")

    def _on_tree_motion(self, event):
        """Muestra tooltip basado en la columna bajo el cursor."""
        region = self.tree_resumen.identify("region", event.x, event.y)
        if region == "heading":
            col_id = self.tree_resumen.identify_column(event.x)
            # col_id es '#1', '#2', etc.
            if not col_id: return
            idx = int(col_id.replace("#", "")) - 1
            cols = self.tree_resumen["columns"]
            if 0 <= idx < len(cols):
                col_name = cols[idx]
                desc = self.cols_desc.get(col_name, col_name)
                self._show_tooltip(event.x_root, event.y_root, f"{col_name}\n{desc}")
        elif region == "cell":
            # Opcional: mostrar valor completo o descripción de columna también
            col_id = self.tree_resumen.identify_column(event.x)
            if not col_id: return
            idx = int(col_id.replace("#", "")) - 1
            cols = self.tree_resumen["columns"]
            if 0 <= idx < len(cols):
                col_name = cols[idx]
                desc = self.cols_desc.get(col_name, "")
                # item = self.tree_resumen.identify_row(event.y)
                # Opcional: obtener valor celda
                self._show_tooltip(event.x_root, event.y_root, f"{col_name}\n{desc}")
        else:
            self._hide_tooltip()

    def _show_tooltip(self, x, y, text):
        if self.tooltip_window:
            # Si ya existe, solo actualizar texto y pos?
            # Para evitar parpadeo, mejor recrear solo si cambia mucho o mover
            pass
            
        self._hide_tooltip()
        
        self.tooltip_window = tk.Toplevel(self.ventana)
        self.tooltip_window.wm_overrideredirect(True)
        self.tooltip_window.wm_geometry(f"+{x+15}+{y+15}")
        
        label = ttk.Label(self.tooltip_window, text=text, background="#ffffe0", relief="solid", borderwidth=1, padding=5)
        label.pack()
        
    def _hide_tooltip(self, event=None):
        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None

    def crear_tab_exportar(self):
        """Crea el tab de exportación."""
        frame_exportar = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(frame_exportar, text="Exportar")
        
        ttk.Label(frame_exportar, text="Exportar resultados:", font=("Arial", 12, "bold")).pack(pady=10)
        
        # Botones de exportación
        ttk.Button(frame_exportar, text="Guardar Mapa como PNG", 
                  command=self.exportar_mapa).pack(pady=5, fill=tk.X, padx=20)
        
        ttk.Button(frame_exportar, text="Exportar Métricas (CSV)", 
                  command=self.exportar_metricas_csv).pack(pady=5, fill=tk.X, padx=20)
        
        ttk.Button(frame_exportar, text="Exportar Métricas (JSON)", 
                  command=self.exportar_metricas_json).pack(pady=5, fill=tk.X, padx=20)
        
        ttk.Button(frame_exportar, text="Generar Reporte HTML", 
                  command=self.exportar_reporte_html).pack(pady=5, fill=tk.X, padx=20)
        
        ttk.Separator(frame_exportar, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        ttk.Button(frame_exportar, text="Generar Sitio Web Completo (Batch)", 
                  command=self.generar_sitio_web).pack(pady=5, fill=tk.X, padx=20)
                  

        
        ttk.Separator(frame_exportar, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=20)
        
        # Información de directorio
        self.label_directorio = ttk.Label(frame_exportar, text="", wraplength=400)
        self.label_directorio.pack(pady=10)
    
    def crear_barra_progreso(self, parent):
        """Crea la barra de estado estilo IDE (una sola línea)."""
        # Frame principal de la barra de estado con borde hundido
        self.frame_estado = ttk.Frame(parent, relief=tk.SUNKEN, padding="2")
        self.frame_estado.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.S), pady=(5, 0))
        
        # --- SECCIÓN IZQUIERDA (Estado y Controles) ---
        
        # 1. Label de Estado
        self.label_estado = ttk.Label(self.frame_estado, text="Listo", width=50)
        self.label_estado.pack(side=tk.LEFT, padx=(5, 10))
        
        # 2. Barra de progreso personalizada (Canvas)
        self.canvas_progreso = tk.Canvas(self.frame_estado, height=20, bg="#f0f0f0", highlightthickness=0)
        # self.canvas_progreso.pack(side=tk.LEFT, padx=(0, 5), fill=tk.X, expand=True) <-- Se mostrará al iniciar
        
        # Variables para animación
        self.progreso_animando = False
        self.progreso_x = 0
        self.progreso_direction = 1
        self.progreso_width = 60
        self.progreso_timer_id = None
        
        # 3. Botón cancelar (compacto, inicialmente oculto)
        self.boton_cancelar = ttk.Button(self.frame_estado, text="⏹", width=3,
                                         command=self.cancelar_analisis,
                                         state='disabled')
        # self.boton_cancelar.pack(side=tk.LEFT, padx=(0, 5)) <-- Se mostrará al iniciar
        
        # --- SECCIÓN DERECHA (Información) ---
        # Frame contenedor para la información (inicialmente oculto)
        self.frame_info_derecha = ttk.Frame(self.frame_estado)
        # No hacemos pack() todavía
        
        # Agregamos elementos de derecha a izquierda dentro del frame
        
        # Función helper para separadores
        def add_separator():
            ttk.Separator(self.frame_info_derecha, orient=tk.VERTICAL).pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=2)
        
        # 7. Archivo (Clickable)
        self.label_ruta = ttk.Label(self.frame_info_derecha, text="—", foreground="blue", cursor="hand2")
        self.label_ruta.pack(side=tk.RIGHT, padx=(5, 10))
        self.label_ruta.bind("<Button-1>", self.abrir_archivo)
        
        ttk.Label(self.frame_info_derecha, text="Archivo:", font=("Arial", 8, "bold")).pack(side=tk.RIGHT, padx=(5, 0))
        
        add_separator()
        
        # 6. Tiempo
        self.label_tiempo = ttk.Label(self.frame_info_derecha, text="—")
        self.label_tiempo.pack(side=tk.RIGHT, padx=(5, 0))
        ttk.Label(self.frame_info_derecha, text="Tiempo:", font=("Arial", 8, "bold")).pack(side=tk.RIGHT, padx=(5, 0))
        
        add_separator()
        
        # 5. Red
        self.label_red_actual = ttk.Label(self.frame_info_derecha, text="—")
        self.label_red_actual.pack(side=tk.RIGHT, padx=(5, 0))
        ttk.Label(self.frame_info_derecha, text="Red:", font=("Arial", 8, "bold")).pack(side=tk.RIGHT, padx=(5, 0))
        
        add_separator()
        
        # 4. Aristas
        self.label_aristas = ttk.Label(self.frame_info_derecha, text="—")
        self.label_aristas.pack(side=tk.RIGHT, padx=(5, 0))
        ttk.Label(self.frame_info_derecha, text="Aristas:", font=("Arial", 8, "bold")).pack(side=tk.RIGHT, padx=(5, 0))
        
        add_separator()
        
        # 3. Nodos
        self.label_nodos = ttk.Label(self.frame_info_derecha, text="—")
        self.label_nodos.pack(side=tk.RIGHT, padx=(5, 0))
        ttk.Label(self.frame_info_derecha, text="Nodos:", font=("Arial", 8, "bold")).pack(side=tk.RIGHT, padx=(5, 0))

    def abrir_archivo(self, event):
        """Abre el archivo o directorio mostrado en el label de ruta."""
        ruta = self.label_ruta.cget("text")
        if ruta and ruta != "—" and os.path.exists(ruta):
            try:
                os.startfile(ruta)
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo abrir el archivo:\n{e}")
    
    def cargar_lista_ciudades(self):
        """Carga la lista de ciudades disponibles según el dataset seleccionado."""
        dataset = self.combo_dataset.get()
        
        if dataset == "Kujala":
            directorio = self.directorio_datos / "kujala" / "procesado"
        else:  # Metro51
            directorio = self.directorio_datos / "51_metro_networks"
        
        if not directorio.exists():
            self.combo_ciudad['values'] = ["(No hay datos disponibles)"]
            self.combo_ciudad.current(0)
            return
        
        # Listar subdirectorios (Kujala) o archivos JSON (Metro51)
        if dataset == "Kujala":
            ciudades = sorted([d.name for d in directorio.iterdir() if d.is_dir()])
        else:
            ciudades = sorted([f.stem for f in directorio.rglob("*.json")])
        
        if ciudades:
            self.combo_ciudad['values'] = ciudades
            self.combo_ciudad.current(0)
        else:
            self.combo_ciudad['values'] = ["(No hay datos disponibles)"]
            self.combo_ciudad.current(0)
    
    def on_dataset_changed(self, event=None):
        """Maneja el cambio de dataset."""
        self.cargar_lista_ciudades()
    
    def generar_analisis(self):
        """Inicia el análisis de la red seleccionada."""
        ciudad = self.combo_ciudad.get()
        if not ciudad or ciudad == "(No hay datos disponibles)":
            messagebox.showwarning("Advertencia", "Por favor seleccione una ciudad/red válida")
            return
        
        # Resetear flag de cancelación
        self.cancelar_procesamiento = False
        
        # Deshabilitar controles
        self.boton_generar.config(state='disabled')
        self.combo_dataset.config(state='disabled')
        self.combo_ciudad.config(state='disabled')
        self.boton_cancelar.config(state='normal')
        
        # Ocultar panel de información si estaba visible
        self.frame_info_derecha.pack_forget()
        
        # Mostrar progreso
        self.mensaje_estado_actual = f"Procesando {ciudad}..."
        self.label_estado.config(text=self.mensaje_estado_actual)
        self.canvas_progreso.pack(side=tk.LEFT, padx=(0, 5), fill=tk.X, expand=True)
        self.boton_cancelar.pack(side=tk.LEFT, padx=(0, 5))
        self.iniciar_animacion_progreso()
        
        self.tiempo_inicio_analisis = time.time()
        
        # Iniciar cronómetro en hilo separado
        self.evento_stop_timer.clear()
        threading.Thread(target=self.ejecutar_cronometro, daemon=True).start()
        
        # Ejecutar en thread separado
        self.thread_actual = threading.Thread(
            target=self.procesar_red_thread,
            args=(ciudad,),
            daemon=True
        )
        self.thread_actual.start()

    def ejecutar_cronometro(self):
        """Hilo dedicado a actualizar el cronómetro cada segundo."""
        while not self.evento_stop_timer.is_set():
            time.sleep(1)
            if self.evento_stop_timer.is_set():
                break
            tiempo_actual = time.time() - self.tiempo_inicio_analisis
            self.cola_resultados.put(('tiempo', tiempo_actual))

    def iniciar_animacion_progreso(self):
        """Inicia la animación de la barra de progreso personalizada."""
        self.progreso_animando = True
        self.progreso_x = 0
        self.progreso_direction = 1
        self.animar_progreso()

    def detener_animacion_progreso(self):
        """Detiene la animación."""
        self.progreso_animando = False
        if self.progreso_timer_id:
            self.ventana.after_cancel(self.progreso_timer_id)
            self.progreso_timer_id = None

    def animar_progreso(self):
        """Dibuja el frame actual de la animación."""
        if not self.progreso_animando:
            return
            
        width = self.canvas_progreso.winfo_width()
        if width <= 1: width = 200 # Fallback inicial
        
        # Limpiar canvas
        self.canvas_progreso.delete("all")
        
        # Dibujar barra de fondo (opcional)
        # self.canvas_progreso.create_rectangle(0, 0, width, 20, fill="#e0e0e0", width=0)
        
        # Dibujar bloque animado
        x1 = self.progreso_x
        x2 = x1 + self.progreso_width
        self.canvas_progreso.create_rectangle(x1, 0, x2, 20, fill="#0078d7", width=0) # Azul Windows
        
        # Actualizar posición
        step = 5
        if self.progreso_direction == 1:
            self.progreso_x += step
            if self.progreso_x + self.progreso_width >= width:
                self.progreso_direction = -1
        else:
            self.progreso_x -= step
            if self.progreso_x <= 0:
                self.progreso_direction = 1
        
        # Dibujar texto de tiempo (SIEMPRE ENCIMA)
        tiempo_texto = self.mensaje_estado_actual.split('(')[-1].replace(')', '') if '(' in self.mensaje_estado_actual else "0s"
        if "Análisis completado" in self.mensaje_estado_actual: tiempo_texto = "" # No mostrar en completado
        
        # Centrar texto
        self.canvas_progreso.create_text(width/2, 10, text=tiempo_texto, fill="black", font=("Arial", 8))
        
        # Programar siguiente frame
        self.progreso_timer_id = self.ventana.after(30, self.animar_progreso)

    def cancelar_analisis(self):
        """Cancela el análisis en curso."""
        self.cancelar_procesamiento = True
        self.label_estado.config(text="Cancelando...")
        self.boton_cancelar.config(state='disabled')

    def procesar_red_thread(self, nombre_ciudad: str):
        """Procesa la red en un thread separado (no bloquea UI)."""
        try:
            inicio = time.time()
            dataset = self.combo_dataset.get()
            ruta_archivo = ""
            
            # 1. Cargar grafo
            if self.cancelar_procesamiento:
                self.cola_resultados.put(('cancelado', 'Operación cancelada por el usuario'))
                return
            
            self.cola_resultados.put(('estado', 'Cargando grafo...'))
            
            if dataset == "Kujala":
                directorio_ciudad = self.directorio_datos / "kujala" / "procesado" / nombre_ciudad
                
                # Intentar apuntar a un archivo específico
                archivo_combined = directorio_ciudad / "network_combined.csv"
                archivo_nodes = directorio_ciudad / "network_nodes.csv"
                
                if archivo_combined.exists():
                    ruta_archivo = str(archivo_combined)
                elif archivo_nodes.exists():
                    ruta_archivo = str(archivo_nodes)
                else:
                    ruta_archivo = str(directorio_ciudad)
                    
                grafo, _ = preparar_redes.construir_grafo_desde_ciudad_kujala(directorio_ciudad)
            else:  # Metro51
                # Buscar archivo JSON
                archivos = list((self.directorio_datos / "51_metro_networks").rglob(f"{nombre_ciudad}.json"))
                if not archivos:
                    raise FileNotFoundError(f"No se encontró {nombre_ciudad}.json")
                
                ruta_archivo = str(archivos[0])
                # Cargar usando la función de metros
                grafos, _ = preparar_redes.cargar_metros_desde_carpeta(
                    archivos[0].parent,
                    usar_paralelo=False
                )
                grafo = grafos.get(nombre_ciudad)
                if grafo is None:
                    raise ValueError(f"No se pudo cargar {nombre_ciudad}")
            
            # 2. Calcular métricas
            # 2. Calcular métricas estandarizadas (18 columnas)
            if self.cancelar_procesamiento:
                self.cola_resultados.put(('cancelado', 'Operación cancelada por el usuario'))
                return
            
            self.cola_resultados.put(('estado', 'Calculando todas las métricas de robustez...'))
            
            # Generar simplificación
            grafo_simplificado = grafo.obtener_topologia_simplificada()
            n_nodos_simp = grafo_simplificado.numero_de_nodos()
            n_aristas_simp = grafo_simplificado.numero_de_aristas()
            
            # Usamos grafos simplificados (Topología) para todas las métricas
            metricas = procesar_redes.calcular_todas_metricas_robustez(grafo_simplificado)
            
            # También calculamos curva robustez para el gráfico (separado del dict de resumen)
            self.cola_resultados.put(('estado', 'Calculando curvas de robustez...'))
            metricas['curva_robustez'] = procesar_redes.calcular_curva_robustez(
                grafo_simplificado, estrategias=["grado", "aleatorio"], pasos=20, semilla=42
            )
            
            # 3. Guardar resultados
            if self.cancelar_procesamiento:
                self.cola_resultados.put(('cancelado', 'Operación cancelada por el usuario'))
                return
            
            self.cola_resultados.put(('estado', 'Guardando resultados...'))
            dir_salida = visualizacion.crear_directorio_salida(
                f"{dataset[0]}_{nombre_ciudad}",
                self.directorio_procesados
            )
            visualizacion.guardar_resultados_red(
                 grafo, metricas, dir_salida, 
                 generar_mapa=False,
                 grafo_metricas=grafo_simplificado
            )
            
            # 4. Enviar resultados a UI
            tiempo_total = time.time() - inicio
            self.cola_resultados.put(('exito', grafo, metricas, dir_salida, tiempo_total, ruta_archivo, n_nodos_simp, n_aristas_simp))
            
        except Exception as e:
            self.cola_resultados.put(('error', str(e)))
    
    def verificar_cola(self):
        """Verifica la cola de resultados del thread."""
        try:
            while True:
                resultado = self.cola_resultados.get_nowait()
                
                if resultado[0] == 'estado':
                    self.mensaje_estado_actual = resultado[1]
                    self.label_estado.config(text=self.mensaje_estado_actual)
                
                elif resultado[0] == 'tiempo':
                    tiempo_actual = resultado[1]
                    # Actualizamos el mensaje de estado para que el loop de animación lo recoja
                    base_msg = self.mensaje_estado_actual.split('(')[0].strip()
                    self.mensaje_estado_actual = f"{base_msg} ({int(tiempo_actual)}s)"
                    # No actualizamos label_estado aquí para evitar parpadeo, solo el overlay en el canvas
                
                elif resultado[0] == 'exito':
                    self.evento_stop_timer.set()  # Detener cronómetro
                    _, grafo, metricas, dir_salida, tiempo_total, ruta_archivo, n_nodos_simp, n_aristas_simp = resultado
                    self.actualizar_ui_con_resultados(grafo, metricas, dir_salida)
                    
                    # Actualizar panel de estado con información de la red
                    # Formato solicitado: Total (Simplificados)
                    self.label_nodos.config(text=f"{grafo.numero_de_nodos()} ({n_nodos_simp})")
                    self.label_aristas.config(text=f"{grafo.numero_de_aristas()} ({n_aristas_simp})")
                    self.label_red_actual.config(text=grafo.nombre)
                    self.label_tiempo.config(text=f"{tiempo_total:.2f} s")
                    self.label_ruta.config(text=ruta_archivo)
                    
                    # Mostrar panel de información
                    self.frame_info_derecha.pack(side=tk.RIGHT, fill=tk.Y)
                    
                    self.detener_animacion_progreso()
                    self.canvas_progreso.pack_forget()
                    self.boton_cancelar.pack_forget()
                    self.boton_generar.config(state='normal')
                    self.combo_dataset.config(state='readonly')
                    self.combo_ciudad.config(state='readonly')
                    self.boton_cancelar.config(state='disabled')
                    mensaje_final = f"Análisis completado en {tiempo_total:.2f} segundos"
                    self.label_estado.config(text=mensaje_final)
                    self.mensaje_estado_actual = mensaje_final
                    messagebox.showinfo("Éxito", f"{mensaje_final}\nResultados guardados en:\n{dir_salida}")
                
                elif resultado[0] == 'error':
                    self.evento_stop_timer.set()  # Detener cronómetro
                    self.detener_animacion_progreso()
                    self.canvas_progreso.pack_forget()
                    self.boton_cancelar.pack_forget()
                    self.boton_generar.config(state='normal')
                    self.combo_dataset.config(state='readonly')
                    self.combo_ciudad.config(state='readonly')
                    self.boton_cancelar.config(state='disabled')
                    self.label_estado.config(text="Error en el análisis")
                    messagebox.showerror("Error", f"Error al procesar la red:\n{resultado[1]}")
                
                elif resultado[0] == 'cancelado':
                    self.evento_stop_timer.set()  # Detener cronómetro
                    self.limpiar_paneles()  # Limpiar paneles al cancelar
                    self.detener_animacion_progreso()
                    self.canvas_progreso.pack_forget()
                    self.boton_cancelar.pack_forget()
                    self.boton_generar.config(state='normal')
                    self.combo_dataset.config(state='readonly')
                    self.combo_ciudad.config(state='readonly')
                    self.boton_cancelar.config(state='disabled')
                    self.label_estado.config(text="Listo")
                    self.label_estado.config(text="Listo")
                    messagebox.showinfo("Cancelado", resultado[1])

                elif resultado[0] == 'batch_exito':
                    self.evento_stop_timer.set()
                    ruta, total = resultado[1], resultado[2]
                    
                    self.limpiar_paneles()
                    self.detener_animacion_progreso()
                    self.canvas_progreso.pack_forget()
                    self.boton_cancelar.pack_forget()
                    
                    self.boton_generar.config(state='normal')
                    self.combo_dataset.config(state='readonly')
                    self.combo_ciudad.config(state='readonly')
                    self.boton_cancelar.config(state='disabled')
                    
                    mensaje = f"Sitio Web generado exitosamente\\n{total} redes procesadas\\nUbicación: {ruta}"
                    self.label_estado.config(text="Sitio web generado")
                    messagebox.showinfo("Proceso Completado", mensaje)
                    
                    # Abrir carpeta
                    try:
                        os.startfile(ruta)
                    except:
                        pass

        except queue.Empty:
            pass
        finally:
            self.ventana.after(100, self.verificar_cola)
    
    def limpiar_paneles(self):
        """Limpia los paneles de mapa y análisis volviendo al estado inicial."""
        # Limpiar panel de mapa
        for widget in self.frame_canvas_mapa.winfo_children():
            widget.destroy()
        
        # Restaurar label inicial del mapa
        self.label_mapa = ttk.Label(self.frame_canvas_mapa, 
                                     text="Seleccione una red y presione 'Generar Análisis'", 
                                     font=("Arial", 12))
        self.label_mapa.pack(expand=True)
        
        # Limpiar tab de info
        for item in self.tree_info.get_children():
            self.tree_info.delete(item)
        
        # Limpiar tab de robustez
        for widget in self.frame_canvas_robustez.winfo_children():
            widget.destroy()
        
        # Limpiar tab de componentes
        for widget in self.frame_canvas_componentes.winfo_children():
            widget.destroy()
        
        # Limpiar label de directorio en tab exportar
        self.label_directorio.config(text="")
        
        # Resetear variables de estado
        self.grafo_actual = None
        self.metricas_actuales = None
    
    def actualizar_ui_con_resultados(self, grafo, metricas, dir_salida):
        """Actualiza la UI con los resultados del análisis."""
        self.grafo_actual = grafo
        self.metricas_actuales = metricas
        self.dir_salida_actual = dir_salida

        
        # 1. Actualizar mapa
        self.actualizar_mapa(grafo, dir_salida)
        
        # 1.5 Programar captura de pantalla y regeneración de reporte
        # Esperamos 4.0s para dar tiempo a que carguen los tiles
        ruta_mapa = dir_salida / "imagenes" / "mapa.png"
        self.ventana.after(4000, lambda: self._post_proceso_mapa(ruta_mapa))



        # 2. Actualizar tab de info
        self.actualizar_tab_info(metricas)
        
        # 3. Actualizar tab de robustez
        self.actualizar_tab_robustez(metricas)
        
        # 4. Actualizar tab de componentes
        self.actualizar_tab_componentes(grafo)
        
        # 5. Actualizar info de directorio
        self.label_directorio.config(text=f"Resultados guardados en:\\n{dir_salida}")
    
    
    def actualizar_mapa(self, grafo, dir_salida):
        for widget in self.frame_canvas_mapa.winfo_children():
            if hasattr(self.mapa_widget, 'map_widget') and self.mapa_widget.map_widget and widget == self.mapa_widget.map_widget:
                continue
            widget.destroy()
        
        if not self.mapa_widget.actualizar_con_grafo(grafo):
            label = ttk.Label(self.frame_canvas_mapa, text=f"No hay datos geográficos para\n{grafo.nombre}", font=("Arial", 12))
            label.pack(expand=True)
            
    def capturar_imagen_mapa(self, ruta_destino: Path):
        """Captura el widget de mapa actual y lo guarda como imagen."""
        if not self.mapa_widget or not self.mapa_widget.map_widget:
            return
            
        try:
            # Forzar actualización de tareas pendientes de la UI
            self.ventana.update_idletasks()
            
            widget = self.mapa_widget.map_widget
            x = widget.winfo_rootx()
            y = widget.winfo_rooty()
            w = widget.winfo_width()
            h = widget.winfo_height()
            
            # Capturar
            img = ImageGrab.grab(bbox=(x, y, x+w, y+h))
            img.save(ruta_destino)
            print(f"[INFO] Mapa capturado desde UI: {ruta_destino}")
        except Exception as e:
            print(f"[ERROR] Al capturar mapa: {e}")
            
    def _post_proceso_mapa(self, ruta_mapa: Path):
        """Captura el mapa y regenera el reporte HTML."""
        self.capturar_imagen_mapa(ruta_mapa)
        
        # Regenerar reporte para incluir la nueva imagen
        if self.grafo_actual and self.metricas_actuales and self.dir_salida_actual:
            ruta_html = self.dir_salida_actual / "reportes" / "reporte.html"
            visualizacion._generar_reporte_individual(
                self.grafo_actual, 
                self.metricas_actuales, 
                ruta_html,
                directorio_origen_imagenes=self.dir_salida_actual
            )



    def regenerar_snapshot_reporte(self):
        """Permite al usuario actualizar la captura del mapa si los tiles no cargaron a tiempo."""
        if not self.dir_salida_actual:
            messagebox.showwarning("Advertencia", "No hay resultados activos para actualizar.")
            return
            
        ruta_mapa = self.dir_salida_actual / "imagenes" / "mapa.png"
        try:
            self._post_proceso_mapa(ruta_mapa)
            messagebox.showinfo("Éxito", "Imagen del mapa y reporte actualizados con la vista actual.")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo actualizar: {e}")

    def cambiar_proveedor_mapa(self, event=None):
        if hasattr(self, 'mapa_widget') and self.mapa_widget:
            self.mapa_widget.cambiar_proveedor(self.combo_proveedor.get())


    def actualizar_tab_info(self, metricas):
        """Actualiza el tab de información general con métricas y grafo."""
        # 1. Actualizar tabla
        for item in self.tree_info.get_children():
            self.tree_info.delete(item)
        
        for clave, valor in metricas.items():
            if isinstance(valor, float):
                # Usar notación científica solo para valores muy pequeños
                if abs(valor) < 0.001 and valor != 0:
                    valor_str = f"{valor:.3e}"
                else:
                    valor_str = f"{valor:.3f}"
            else:
                valor_str = str(valor)
            self.tree_info.insert("", tk.END, values=(clave, valor_str))
            
        # 2. Actualizar grafo topológico
        # 2. Actualizar grafo topológico
        if self.grafo_actual:
            self.grafo_interactivo.dibujar_grafo(self.grafo_actual)
            
        # 3. Actualizar Gráfico Radar
        self.actualizar_grafico_radar(metricas)

    def actualizar_grafico_radar(self, metricas):
        """Genera el gráfico radial de robustez con normalización global."""
        # Limpiar canvas anterior
        for widget in self.frame_radar.winfo_children():
            widget.destroy()
            
        # Definir claves internas y mapeo a columnas CSV (Colab / Paper)
        # Claves internas (GUI/Backend) -> Columnas CSV (Global Stats)
        # Orden del Colab: Avg_Deg, Nat_Conn, Kappa, r_T, Clustering, Meshedness, C_G, Rel_G, Efficiency, Alg_Conn
        
        # Mapeo de nuestras claves retornadas por backend a columnas del CSV
        # Backend returns: Ē[D], λ̄*, Kappa, r̄ᵀ, CC_G, M_G, C_G, Rel_G, E[1/H], μ̄ₙ₋₁
        # CSV columns:     Avg_Deg, Nat_Conn, Kappa, r_T, Clustering, Meshedness, C_G, Rel_G, Efficiency, Alg_Conn
        
        map_gui_to_csv = {
            "Ē[D]": "Avg_Deg",
            "λ̄*": "Nat_Conn",
            "Kappa": "Kappa", # Nuevo: Usamos Kappa directament
            "r̄ᵀ": "r_T",
            "CC_G": "Clustering",
            "M_G": "Meshedness",
            "C_G": "C_G",
            "Rel_G": "Rel_G",
            "E[1/H]": "Efficiency",
            "μ̄ₙ₋₁": "Alg_Conn"
        }

        # Orden de visualización (según Colab)
        orden_keys = ["Ē[D]", "λ̄*", "Kappa", "r̄ᵀ", "CC_G", "M_G", "C_G", "Rel_G", "E[1/H]", "μ̄ₙ₋₁"]
        
        # Etiquetas (según Colab)
        labels_display = [
            r"$\overline{E[D]}$", r"$\overline{\lambda}^*$", r"$\kappa$", 
            r"${r^T}$", r"$CC_G$", r"$M_G$", 
            r"$C_G$", r"$Rel_G$", r"$E[\frac{1}{H}]$", r"$\overline{\mu_{N-1}}$"
        ]
        
        # Cargar datos globales para normalización
        try:
            df_global = pd.read_csv("utils/resultados_metro_robustez_completo.csv")
        except Exception:
            df_global = None

        valores = []
        for k in orden_keys:
            val = metricas.get(k, 0.0)
            if not isinstance(val, (int, float)):
                val = 0.0
            
            # Normalización Global
            col_csv = map_gui_to_csv.get(k)
            if df_global is not None and col_csv in df_global.columns:
                min_val = df_global[col_csv].min()
                max_val = df_global[col_csv].max()
                
                # Caso especial: Si el valor actual está fuera de rango (ej: nueva red), clamp?
                # O recalculamos min/max incluyendo el valor actual? 
                # El Colab usa el dataset completo. Si estamos viendo una red QUE ESTÁ en el dataset, todo bien.
                # Si es una red nueva, deberíamos idealmente proyectarla en el rango existente o actualizarlo.
                # Por simplicidad y consistencia visual con el "Atlas", usamos el rango del dataset fijo.
                
                # Update bounds with current value just in case
                min_val = min(min_val, val)
                max_val = max(max_val, val)
                
                if max_val - min_val > 0:
                    val_norm = (val - min_val) / (max_val - min_val)
                else:
                    val_norm = 0.5
                valores.append(val_norm)
            else:
                # Fallback si no hay CSV: asumir 0-1 si parece estar en rango, o dejar raw?
                # La mayoría de métricas NO son 0-1 (Kappa, Avg_Deg). 
                # Sin CSV, el gráfico se verá mal.
                valores.append(val if val <= 1.0 else 1.0) # Clipping simple fallback
        
        # Cerrar el polígono
        valores_plot = valores + [valores[0]]
        
        # Configurar ángulos
        num_vars = len(orden_keys)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += [angles[0]]  # Cerrar loop
        
        fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
        
        # Configurar dirección (Horario, inicio arriba)
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        
        # Dibujar líneas y relleno
        ax.plot(angles, valores_plot, linewidth=2, linestyle='solid', color='#1f77b4') # Color Colab
        ax.fill(angles, valores_plot, 'b', alpha=0.3, color='#1f77b4')
        
        # Etiquetas
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels_display, fontsize=10, fontweight='bold')
        
        # Ajustar grid y límites
        ax.set_ylim(0, 1)
        ax.set_rlabel_position(0)
        # Quitar etiquetas radiales internas para limpiar como en Colab
        ax.set_yticklabels([]) 
        ax.set_yticks([])
        
        # Título con nombre de ciudad
        ciudad = metricas.get("Metros", "Red")
        if "argentina" in ciudad.lower(): ciudad = "Buenos Aires" 
        else: ciudad = ciudad.replace("-", " ").title()
        
        ax.set_title(ciudad, va='bottom', fontweight='bold', fontsize=14, y=1.08)
        
        plt.tight_layout()
        
        self.fig_radar = fig # Guardar referencia para exportación
        
        # Embeber
        canvas = FigureCanvasTkAgg(fig, master=self.frame_radar)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    
    def actualizar_tab_robustez(self, metricas):
        """Actualiza el tab de robustez con gráficos."""
        # Limpiar canvas anterior
        for widget in self.frame_canvas_robustez.winfo_children():
            widget.destroy()
        
        # Crear figura con subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))
        fig.suptitle('Métricas de Robustez', fontsize=16, fontweight='bold')
        
        # 1. Indicadores teóricos (barras)
        # 1. Indicadores teóricos (barras)
        # Claves nuevas: r̄ᵀ, C_G, Rel_G
        keys_teoricas = ['r̄ᵀ', 'C_G', 'Rel_G']
        vals_teoricas = []
        labels_teoricas = []
        
        for k in keys_teoricas:
            if k in metricas:
                vals_teoricas.append(metricas[k])
                labels_teoricas.append(k)
        
        if vals_teoricas:
            colores = ['#2ecc71', '#3498db', '#9b59b6']
            # Asegurar que colores alcance
            colores = colores[:len(vals_teoricas)]
            
            bars = ax1.bar(labels_teoricas, vals_teoricas, color=colores, alpha=0.7, edgecolor='black')
            ax1.set_ylabel('Valor Normalizado', fontsize=10)
            ax1.set_title('Indicadores Teóricos', fontsize=12, fontweight='bold')
            ax1.set_ylim(0, max(vals_teoricas) * 1.2 if vals_teoricas else 1)
            ax1.grid(axis='y', alpha=0.3)
            
            # Agregar valores sobre las barras
            for bar, val in zip(bars, vals_teoricas):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.3f}',
                        ha='center', va='bottom', fontsize=9)
        else:
            ax1.text(0.5, 0.5, 'No hay datos\ndisponibles', 
                    ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Indicadores Teóricos', fontsize=12, fontweight='bold')
        
        # 2. Curvas de Robustez (Líneas)
        if 'curva_robustez' in metricas:
            curvas = metricas['curva_robustez']
            
            # Random Failure
            if 'aleatorio' in curvas:
                x = curvas['aleatorio']['x']
                y = curvas['aleatorio']['y']
                ax2.plot(x, y, 'o-', color='black', markerfacecolor='none', label='Random Failure', linewidth=1, markersize=4)
            
            # Targeted Attack
            if 'grado' in curvas:
                x = curvas['grado']['x']
                y = curvas['grado']['y']
                ax2.plot(x, y, 'o-', color='red', label='Targeted Attack', linewidth=1, markersize=4)
            
            ax2.set_xlabel('Number of Nodes Removed', fontsize=10)
            ax2.set_ylabel('Size of Largest Connected Component', fontsize=10)
            ax2.set_title('Robustness Analysis', fontsize=12, fontweight='bold')
            ax2.grid(True, linestyle='--', alpha=0.7)
            ax2.legend(loc='best', frameon=True, fancybox=True, framealpha=0.9)
            
            # Línea punteada de referencia (opcional, ej: 90% del tamaño original)
            # ax2.axhline(y=metricas['n_nodos']*0.9, color='black', linestyle=':', label='f_90%')
            
        elif 'robustez_grado_20pct' in metricas and 'robustez_aleatorio_20pct' in metricas:
            # Fallback a gráfico de barras si no hay curvas calculadas (compatibilidad)
            estrategias = ['Por Grado\n(20%)', 'Aleatoria\n(20%)']
            valores_rob = [metricas['robustez_grado_20pct'], metricas['robustez_aleatorio_20pct']]
            colores_rob = ['#e74c3c', '#f39c12']
            
            bars = ax2.bar(estrategias, valores_rob, color=colores_rob, alpha=0.7, edgecolor='black')
            ax2.set_ylabel('Fracción de nodos\nen componente gigante', fontsize=9)
            ax2.set_title('Robustez por Remoción', fontsize=12, fontweight='bold')
            ax2.set_ylim(0, 1.1)
            ax2.grid(axis='y', alpha=0.3)
            
            for bar, val in zip(bars, valores_rob):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.3f}',
                        ha='center', va='bottom', fontsize=9)
        else:
            ax2.text(0.5, 0.5, 'No hay datos\ndisponibles', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Robustez por Remoción', fontsize=12, fontweight='bold')
        
        # 3. Explicación de r̄ᵀ
        ax3.axis('off')
        texto_rT = [
            "Indicador r̄ᵀ (Robustez Topológica):",
            "",
            "• Fórmula: r̄ᵀ = (L - N + 1) / N",
            "  (Normalizado)",
            "",
            "• Mide densidad de ciclos",
            "• Altos (>0.1) → Muy robusta",
            "• Bajos (~0) → Tipo árbol (frágil)",
        ]
        if 'r̄ᵀ' in metricas:
            texto_rT.append("")
            texto_rT.append(f"Valor actual: {metricas['r̄ᵀ']:.4f}")
        
        ax3.text(0.05, 0.95, '\n'.join(texto_rT), 
                transform=ax3.transAxes, fontsize=9,
                verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        # 4. Explicación de Rel_G / C_G
        ax4.axis('off')
        texto_CG = [
            "Fiabilidad Rel_G:",
            "• Probabilidad de conexión tras fallos aleatorios.",
            "",
            "Conductancia C_G:",
            "• Facilidad de flujo y redundancia.",
        ]
        if 'Rel_G' in metricas:
            texto_CG.append(f"Rel_G: {metricas['Rel_G']:.4f}")
        if 'C_G' in metricas:
            texto_CG.append(f"C_G:   {metricas['C_G']:.4f}")
        
        ax4.text(0.05, 0.95, '\n'.join(texto_CG), 
                transform=ax4.transAxes, fontsize=9,
                verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        plt.tight_layout()
        
        self.fig_robustez = fig # Guardar referencia para exportación
        
        # Embeber en tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.frame_canvas_robustez)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def actualizar_tab_componentes(self, grafo):
        """Actualiza el tab de componentes con gráficos mejorados."""
        # Limpiar canvas anterior
        for widget in self.frame_canvas_componentes.winfo_children():
            widget.destroy()
        
        componentes = grafo.componentes_conectados()
        
        # Crear figura con subplots y más espacio
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
        fig.suptitle('Análisis de Componentes Conectados', fontsize=14, fontweight='bold', y=0.98)
        
        if componentes:
            # Ordenar por tamaño
            componentes_ordenados = sorted(componentes, key=len, reverse=True)
            
            # 1. Gráfico de barras (top 10 componentes)
            top_n = min(10, len(componentes_ordenados))
            tamanos = [len(comp) for comp in componentes_ordenados[:top_n]]
            etiquetas = [f'C{i+1}' for i in range(top_n)]
            colores = plt.cm.viridis(np.linspace(0, 0.9, top_n))
            
            bars = ax1.barh(etiquetas[::-1], tamanos[::-1], color=colores[::-1], 
                           alpha=0.8, edgecolor='black')
            ax1.set_xlabel('Número de nodos', fontsize=10)
            ax1.set_ylabel('Componente', fontsize=10)
            ax1.set_title(f'Top {top_n} Componentes por Tamaño', fontsize=11, fontweight='bold')
            ax1.grid(axis='x', alpha=0.3)
            
            # Agregar valores en las barras (más compactos)
            for i, (bar, tam) in enumerate(zip(bars, tamanos[::-1])):
                porcentaje = tam / grafo.numero_de_nodos() * 100
                ax1.text(tam, bar.get_y() + bar.get_height()/2,
                        f' {tam} ({porcentaje:.1f}%)',
                        va='center', fontsize=8)
            
            # 2. Gráfico de pastel (distribución) - MEJORADO
            if len(componentes_ordenados) > 1:
                # Mostrar top 5 + "otros"
                top_5 = min(5, len(componentes_ordenados))
                tamanos_pie = [len(comp) for comp in componentes_ordenados[:top_5]]
                
                # Labels más compactos
                labels_pie = [f'C{i+1}' for i in range(top_5)]
                
                if len(componentes_ordenados) > top_5:
                    otros = sum(len(comp) for comp in componentes_ordenados[top_5:])
                    tamanos_pie.append(otros)
                    labels_pie.append(f'Otros')
                
                colores_pie = plt.cm.Set3(np.linspace(0, 1, len(tamanos_pie)))
                
                # Usar labels fuera del gráfico para evitar sobreposición
                def make_autopct(values):
                    def my_autopct(pct):
                        total = sum(values)
                        val = int(round(pct*total/100.0))
                        return f'{pct:.1f}%\n({val})'
                    return my_autopct
                
                wedges, texts, autotexts = ax2.pie(
                    tamanos_pie, 
                    labels=None,  # Sin labels en el gráfico
                    autopct=make_autopct(tamanos_pie),
                    colors=colores_pie,
                    startangle=90,
                    pctdistance=0.85,
                    textprops={'fontsize': 8}
                )
                
                # Mejorar legibilidad de porcentajes
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
                    autotext.set_fontsize(8)
                
                # Agregar leyenda fuera del gráfico
                legend_labels = []
                for i, (label, tam) in enumerate(zip(labels_pie, tamanos_pie)):
                    pct = tam / grafo.numero_de_nodos() * 100
                    if label.startswith('C'):
                        legend_labels.append(f'{label}: {tam} nodos ({pct:.1f}%)')
                    else:
                        legend_labels.append(f'{label}: {tam} nodos ({pct:.1f}%)')
                
                ax2.legend(legend_labels, loc='center left', bbox_to_anchor=(1, 0.5), 
                          fontsize=8, frameon=True)
                ax2.set_title('Distribución de Nodos', fontsize=11, fontweight='bold')
            else:
                # Solo un componente
                ax2.text(0.5, 0.5, f'Red completamente conectada\n{grafo.numero_de_nodos()} nodos',
                        ha='center', va='center', transform=ax2.transAxes,
                        fontsize=12, fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
                ax2.set_title('Distribución de Nodos', fontsize=11, fontweight='bold')
            
            # Información adicional (más compacta)
            info_text = f"Componentes: {len(componentes)} | "
            info_text += f"Conectada: {'Sí' if grafo.esta_conectado() else 'No'} | "
            info_text += f"Gigante: {len(componentes_ordenados[0])} nodos "
            info_text += f"({len(componentes_ordenados[0])/grafo.numero_de_nodos()*100:.1f}%)"
            
            fig.text(0.5, 0.02, info_text, ha='center', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            # No hay componentes
            ax1.text(0.5, 0.5, 'No hay componentes\nen el grafo',
                    ha='center', va='center', transform=ax1.transAxes,
                    fontsize=12)
            ax2.text(0.5, 0.5, 'No hay componentes\nen el grafo',
                    ha='center', va='center', transform=ax2.transAxes,
                    fontsize=12)
        
        plt.tight_layout(rect=[0, 0.05, 0.95, 0.96])  # Ajustar para leyenda
        
        # Embeber en tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.frame_canvas_componentes)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def exportar_mapa(self):
        """Exporta el mapa como PNG."""
        if self.grafo_actual is None:
            messagebox.showwarning("Advertencia", "Primero debe generar un análisis")
            return
        
        archivo = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("Todos", "*.*")]
        )
        
        if archivo:
            # Generar mapa
            fig = visualizacion.generar_mapa_geografico(self.grafo_actual, Path(archivo))
            if fig:
                plt.close(fig)
                messagebox.showinfo("Éxito", f"Mapa guardado en:\\n{archivo}")
            else:
                messagebox.showwarning("Advertencia", "No hay datos geográficos para exportar")
    
    def exportar_metricas_csv(self):
        """Exporta métricas a CSV."""
        if self.metricas_actuales is None:
            messagebox.showwarning("Advertencia", "Primero debe generar un análisis")
            return
        
        archivo = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("Todos", "*.*")]
        )
        
        if archivo:
            df = pd.DataFrame([self.metricas_actuales])
            df.to_csv(archivo, index=False)
            messagebox.showinfo("Éxito", f"Métricas guardadas en:\\n{archivo}")
    
    def exportar_metricas_json(self):
        """Exporta métricas a JSON."""
        if self.metricas_actuales is None:
            messagebox.showwarning("Advertencia", "Primero debe generar un análisis")
            return
        
        archivo = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("Todos", "*.*")]
        )
        
        if archivo:
            import json
            with open(archivo, 'w', encoding='utf-8') as f:
                json.dump(self.metricas_actuales, f, indent=2, ensure_ascii=False, default=str)
            messagebox.showinfo("Éxito", f"Métricas guardadas en:\\n{archivo}")
    
    def exportar_reporte_html(self):
        """Genera reporte HTML."""
        if self.grafo_actual is None or self.metricas_actuales is None:
            messagebox.showwarning("Advertencia", "Primero debe generar un análisis")
            return
            
        # Asegurar que los gráficos existan antes de exportar
        if not hasattr(self, 'fig_radar') or self.fig_radar is None:
            print("[INFO] Generando gráfico radar para exportación...")
            self.actualizar_grafico_radar(self.metricas_actuales)
            
        if not hasattr(self, 'fig_robustez') or self.fig_robustez is None:
            print("[INFO] Generando gráfico robustez para exportación...")
            self.actualizar_tab_robustez(self.metricas_actuales)
        
        archivo = filedialog.asksaveasfilename(
            defaultextension=".html",
            filetypes=[("HTML", "*.html"), ("Todos", "*.*")]
        )
        
        archivo_path = Path(archivo)
        dir_imagenes = archivo_path.parent / "imagenes"
        dir_imagenes.mkdir(parents=True, exist_ok=True)
        
        ruta_radar = dir_imagenes / "radar.png"
        ruta_robustez = dir_imagenes / "robustez.png"
        
        # Guardar imágenes
        try:
            if hasattr(self, 'fig_radar') and self.fig_radar:
                self.fig_radar.savefig(ruta_radar, dpi=100, bbox_inches='tight')
            if hasattr(self, 'fig_robustez') and self.fig_robustez:
                self.fig_robustez.savefig(ruta_robustez, dpi=100, bbox_inches='tight')
        except Exception as e:
            print(f"[WARN] Error guardando imagenes reporte: {e}")
        
        if archivo:
            visualizacion._generar_reporte_individual(
                self.grafo_actual,
                self.metricas_actuales,
                archivo_path,
                directorio_origen_imagenes=self.dir_salida_actual,
                ruta_img_radar=ruta_radar if ruta_radar.exists() else None,
                ruta_img_robustez=ruta_robustez if ruta_robustez.exists() else None
            )
            messagebox.showinfo("Éxito", f"Reporte generado en:\\n{archivo}")

    def generar_sitio_web(self):
        """Inicia la generación del sitio web completo en batch."""
        if messagebox.askyesno("Confirmar", "¿Desea procesar TODAS las redes y generar el sitio web?\\nEsto puede tardar unos minutos."):
            # Resetear estado
            self.cancelar_procesamiento = False
            self.boton_generar.config(state='disabled')
            self.combo_dataset.config(state='disabled')
            self.combo_ciudad.config(state='disabled')
            self.boton_cancelar.config(state='normal')
            
            # Mostrar progreso
            self.label_estado.config(text="Iniciando procesamiento batch...")
            self.canvas_progreso.pack(side=tk.LEFT, padx=(0, 5), fill=tk.X, expand=True)
            self.boton_cancelar.pack(side=tk.LEFT, padx=(0, 5))
            self.iniciar_animacion_progreso()
            
            self.tiempo_inicio_analisis = time.time()
            self.evento_stop_timer.clear()
            threading.Thread(target=self.ejecutar_cronometro, daemon=True).start()
            
            threading.Thread(
                target=self.ejecutar_generacion_sitio_web_thread,
                daemon=True
            ).start()

    def ejecutar_generacion_sitio_web_thread(self):
        """Lógica del thread para generar el sitio web completo."""
        try:
            inicio = time.time()
            grafos_originales = {}
            grafos_simplificados = {}
            
            # 1. Identificar archivos
            # Metro51
            ruta_metro51 = self.directorio_datos / "51_metro_networks"
            archivos_metro = list(ruta_metro51.rglob("*.json")) if ruta_metro51.exists() else []
            
            # Kujala
            ruta_kujala = self.directorio_datos / "kujala" / "procesado"
            dirs_kujala = [d for d in ruta_kujala.iterdir() if d.is_dir()] if ruta_kujala.exists() else []
            
            total_redes = len(archivos_metro) + len(dirs_kujala)
            procesados = 0
            
            # 2. Cargar y simplificar Metro51
            if archivos_metro:
                grafos_m, _ = preparar_redes.cargar_metros_desde_carpeta(ruta_metro51, usar_paralelo=False)
                for name, G in grafos_m.items():
                    if self.cancelar_procesamiento:
                        self.cola_resultados.put(('cancelado', 'Generación de sitio cancelada'))
                        return
                    
                    procesados += 1
                    self.cola_resultados.put(('estado', f"Procesando Metro51 ({procesados}/{total_redes}): {name}"))
                    
                    nombre_final = f"M_{name}"
                    grafos_originales[nombre_final] = G
                    grafos_simplificados[nombre_final] = G.obtener_topologia_simplificada()

            # 3. Cargar y simplificar Kujala
            for d in dirs_kujala:
                if self.cancelar_procesamiento:
                    self.cola_resultados.put(('cancelado', 'Generación de sitio cancelada'))
                    return
                
                procesados += 1
                name = d.name
                self.cola_resultados.put(('estado', f"Procesando Kujala ({procesados}/{total_redes}): {name}"))
                
                try:
                    G, _ = preparar_redes.construir_grafo_desde_ciudad_kujala(d)
                    nombre_final = f"K_{name}"
                    grafos_originales[nombre_final] = G
                    grafos_simplificados[nombre_final] = G.obtener_topologia_simplificada()
                except Exception as e:
                    print(f"Error cargando {name}: {e}")

            if not grafos_originales:
                self.cola_resultados.put(('error', "No se encontraron redes para procesar"))
                return

            # 4. Calcular métricas globales (Fase 3)
            self.cola_resultados.put(('estado', "Calculando métricas globales..."))
            resumen = procesar_redes.calcular_resumen_dataset(
                grafos=grafos_simplificados,
                fraccion_remover=0.2,
                ejecuciones_aleatorias=10,
                semilla=42
            )
            
            # 5. Generar sitio (Fase 4)
            self.cola_resultados.put(('estado', "Generando páginas HTML..."))
            base_sitio = Path("sitio")
            dir_analisis = base_sitio / "analisis"
            dir_analisis.mkdir(parents=True, exist_ok=True)
            
            from hmi.mapa import MapaWidget # Ensure import if needed generally, usually at top
            
            # Copiar assets si es necesario? visualizacion lo maneja?
            
            # Generar reporte HTML global
            visualizacion.generar_reporte_html(
                grafos=grafos_originales,
                df_resumen=resumen,
                ruta_salida=base_sitio / "index.html" # Root index
            )
            
            total_generar = len(resumen)
            count = 0
            for _, row in resumen.iterrows():
                if self.cancelar_procesamiento:
                    self.cola_resultados.put(('cancelado', 'Generación de sitio cancelada'))
                    return
                
                count += 1
                nombre = row['nombre']
                self.cola_resultados.put(('estado', f"Generando reporte ({count}/{total_generar}): {nombre}"))
                
                if nombre in grafos_originales:
                    grafo = grafos_originales[nombre]
                    grafo_simp = grafos_simplificados[nombre]
                    metricas = row.to_dict()
                    
                    dir_red = dir_analisis / nombre
                    dir_red.mkdir(parents=True, exist_ok=True)
                    (dir_red / "imagenes").mkdir(exist_ok=True)
                    (dir_red / "datos").mkdir(exist_ok=True)
                    
                    visualizacion.guardar_resultados_red(
                        grafo=grafo,
                        metricas=metricas,
                        directorio_salida=dir_red,
                        generar_mapa=True,
                        nombre_html="index.html",
                        grafo_metricas=grafo_simp
                    )
            
            self.cola_resultados.put(('batch_exito', base_sitio.resolve(), len(grafos_originales)))
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.cola_resultados.put(('error', f"Error en proceso batch: {str(e)}"))

    def generar_tabla_resumen_batch(self):
        """Genera una tabla HTML con resumen de todas las redes."""
        if hasattr(self, 'procesando_batch') and self.procesando_batch:
             messagebox.showwarning("Busy", "Ya hay un proceso en ejecución.")
             return
             
        # Preguntar ubicación
        path_salida = filedialog.asksaveasfilename(
            defaultextension=".html",
            filetypes=[("HTML File", "*.html")],
            initialfile="tabla_metricas.html",
            title="Guardar Tabla Resumen"
        )
        if not path_salida:
            return
            
        self.procesando_batch = True
        self.boton_generar.config(state='disabled')
        self.boton_cancelar.config(state='normal')
        self.cancelar_procesamiento = False
        
        # UI
        self.label_estado.config(text="Cargando redes para tabla resumen...")
        self.canvas_progreso.pack(side=tk.LEFT, padx=(0, 5), fill=tk.X, expand=True)
        self.boton_cancelar.pack(side=tk.LEFT, padx=(0, 5))
        self.iniciar_animacion_progreso()
        
        # Thread
        threading.Thread(target=self._thread_generar_tabla, args=(path_salida,), daemon=True).start()

    def _thread_generar_tabla(self, path_salida):
        try:
            inicio = time.time()
            grafos = {}
            
            # Cargar Kujala
            dir_kujala = self.directorio_datos / "kujala" / "procesado"
            if dir_kujala.exists():
                for d in dir_kujala.iterdir():
                     if self.cancelar_procesamiento: break
                     if d.is_dir():
                         self.cola_resultados.put(('estado', f'Cargando {d.name}...'))
                         try:
                             G, _ = preparar_redes.construir_grafo_desde_ciudad_kujala(d)
                             grafos[G.nombre] = G
                         except Exception as e:
                             print(f"Skip {d.name}: {e}")
            
            # Cargar Metro51
            dir_metro = self.directorio_datos / "51_metro_networks"
            if dir_metro.exists():
                 archivos = list(dir_metro.rglob("*.json"))
                 if archivos:
                     self.cola_resultados.put(('estado', 'Cargando Metro51...'))
                     g_metros, _ = preparar_redes.cargar_metros_desde_carpeta(dir_metro, usar_paralelo=False)
                     grafos.update(g_metros)
            
            if not grafos:
                self.cola_resultados.put(('error', "No se encontraron redes."))
                return

            if self.cancelar_procesamiento:
                self.cola_resultados.put(('cancelado', "Cancelado por usuario"))
                return
                
            self.cola_resultados.put(('estado', f'Calculando métricas para {len(grafos)} redes...'))
            
            # Calcular resumen (usa paralelo internamente en procesar_redes que ya usa nuestra func actualizada)
            df = procesar_redes.calcular_resumen_dataset(grafos, usar_paralelo=True)
            
            if self.cancelar_procesamiento:
                self.cola_resultados.put(('cancelado', "Cancelado"))
                return
                
            self.cola_resultados.put(('estado', 'Generando HTML...'))
            visualizacion.generar_tabla_resumen_html(df, Path(path_salida))
            
            tiempo = time.time() - inicio
            # Usamos batch_exito para notificar
            self.cola_resultados.put(('batch_exito', str(path_salida), len(grafos)))
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.cola_resultados.put(('error', f"Error generando tabla: {str(e)}"))
        finally:
            self.procesando_batch = False
    
    def ejecutar(self):
        """Ejecuta el loop principal de la aplicación."""
        self.ventana.mainloop()


def main():
    """Función principal para ejecutar la GUI."""
    app = VentanaPrincipal()
    app.ejecutar()


if __name__ == "__main__":
    main()
