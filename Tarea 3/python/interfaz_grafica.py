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

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import pandas as pd
import numpy as np

# Importar módulos del proyecto
import preparar_redes
import procesar_redes
import visualizacion


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
        self.figura_mapa: Optional[plt.Figure] = None
        
        # Cola para comunicación con threads
        self.cola_resultados = queue.Queue()
        
        # Flag para cancelar procesamiento
        self.cancelar_procesamiento = False
        self.thread_actual = None
        
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
        frame_mapa.rowconfigure(0, weight=1)
        
        # Canvas para matplotlib
        self.frame_canvas_mapa = ttk.Frame(frame_mapa)
        self.frame_canvas_mapa.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Label inicial
        self.label_mapa = ttk.Label(self.frame_canvas_mapa, text="Seleccione una red y presione 'Generar Análisis'", 
                                     font=("Arial", 12))
        self.label_mapa.pack(expand=True)
        
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
        
        # Tab 4: Exportar
        self.crear_tab_exportar()
    
    def crear_tab_info(self):
        """Crea el tab de información general."""
        frame_info = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(frame_info, text="Info General")
        
        # TreeView para mostrar métricas
        columns = ("Métrica", "Valor")
        self.tree_info = ttk.Treeview(frame_info, columns=columns, show="headings", height=15)
        self.tree_info.heading("Métrica", text="Métrica")
        self.tree_info.heading("Valor", text="Valor")
        self.tree_info.column("Métrica", width=250)
        self.tree_info.column("Valor", width=200)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(frame_info, orient=tk.VERTICAL, command=self.tree_info.yview)
        self.tree_info.configure(yscroll=scrollbar.set)
        
        self.tree_info.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        frame_info.columnconfigure(0, weight=1)
        frame_info.rowconfigure(0, weight=1)
    
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
        
        ttk.Separator(frame_exportar, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=20)
        
        # Información de directorio
        self.label_directorio = ttk.Label(frame_exportar, text="", wraplength=400)
        self.label_directorio.pack(pady=10)
    
    def crear_barra_progreso(self, parent):
        """Crea la barra de progreso con botón de cancelar."""
        frame_progreso = ttk.Frame(parent, padding="5")
        frame_progreso.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        self.label_estado = ttk.Label(frame_progreso, text="Listo")
        self.label_estado.grid(row=0, column=0, sticky=tk.W)
        
        self.progreso = ttk.Progressbar(frame_progreso, mode='indeterminate')
        self.progreso.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # Botón de cancelar (inicialmente oculto)
        self.boton_cancelar = ttk.Button(frame_progreso, text="Cancelar", 
                                         command=self.cancelar_analisis,
                                         state='disabled')
        self.boton_cancelar.grid(row=1, column=1, padx=(10, 0))
        
        frame_progreso.columnconfigure(0, weight=1)
    
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
        
        # Mostrar progreso
        self.label_estado.config(text=f"Procesando {ciudad}...")
        self.progreso.start()
        
        # Ejecutar en thread separado
        self.thread_actual = threading.Thread(
            target=self.procesar_red_thread,
            args=(ciudad,),
            daemon=True
        )
        self.thread_actual.start()
    
    def cancelar_analisis(self):
        """Cancela el análisis en curso."""
        self.cancelar_procesamiento = True
        self.label_estado.config(text="Cancelando...")
        self.boton_cancelar.config(state='disabled')
    
    def procesar_red_thread(self, nombre_ciudad: str):
        """Procesa la red en un thread separado (no bloquea UI)."""
        try:
            dataset = self.combo_dataset.get()
            
            # 1. Cargar grafo
            if self.cancelar_procesamiento:
                self.cola_resultados.put(('cancelado', 'Operación cancelada por el usuario'))
                return
            
            self.cola_resultados.put(('estado', 'Cargando grafo...'))
            
            if dataset == "Kujala":
                directorio_ciudad = self.directorio_datos / "kujala" / "procesado" / nombre_ciudad
                grafo, _ = preparar_redes.construir_grafo_desde_ciudad_kujala(directorio_ciudad)
            else:  # Metro51
                # Buscar archivo JSON
                archivos = list((self.directorio_datos / "51_metro_networks").rglob(f"{nombre_ciudad}.json"))
                if not archivos:
                    raise FileNotFoundError(f"No se encontró {nombre_ciudad}.json")
                
                # Cargar usando la función de metros
                grafos, _ = preparar_redes.cargar_metros_desde_carpeta(
                    archivos[0].parent,
                    usar_paralelo=False
                )
                grafo = grafos.get(nombre_ciudad)
                if grafo is None:
                    raise ValueError(f"No se pudo cargar {nombre_ciudad}")
            
            # 2. Calcular métricas
            if self.cancelar_procesamiento:
                self.cola_resultados.put(('cancelado', 'Operación cancelada por el usuario'))
                return
            
            self.cola_resultados.put(('estado', 'Calculando métricas...'))
            metricas = procesar_redes.calcular_metricas_basicas(grafo)
            
            if self.cancelar_procesamiento:
                self.cola_resultados.put(('cancelado', 'Operación cancelada por el usuario'))
                return
            
            metricas['r_T'] = procesar_redes.indicador_robustez_rT(grafo)
            
            if self.cancelar_procesamiento:
                self.cola_resultados.put(('cancelado', 'Operación cancelada por el usuario'))
                return
            
            metricas['C_G'] = procesar_redes.conductancia_efectiva_grafo_CG(grafo)
            
            if self.cancelar_procesamiento:
                self.cola_resultados.put(('cancelado', 'Operación cancelada por el usuario'))
                return
            
            self.cola_resultados.put(('estado', 'Calculando robustez por grado...'))
            metricas['robustez_grado_20pct'] = procesar_redes.indice_robustez_simple(
                grafo, fraccion_remover=0.2, estrategia="grado"
            )
            
            if self.cancelar_procesamiento:
                self.cola_resultados.put(('cancelado', 'Operación cancelada por el usuario'))
                return
            
            self.cola_resultados.put(('estado', 'Calculando robustez aleatoria...'))
            metricas['robustez_aleatorio_20pct'] = procesar_redes.indice_robustez_simple(
                grafo, fraccion_remover=0.2, estrategia="aleatorio", semilla=42
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
            visualizacion.guardar_resultados_red(grafo, metricas, dir_salida)
            
            # 4. Enviar resultados a UI
            self.cola_resultados.put(('exito', grafo, metricas, dir_salida))
            
        except Exception as e:
            self.cola_resultados.put(('error', str(e)))
    
    def verificar_cola(self):
        """Verifica la cola de resultados del thread."""
        try:
            while True:
                resultado = self.cola_resultados.get_nowait()
                
                if resultado[0] == 'estado':
                    self.label_estado.config(text=resultado[1])
                
                elif resultado[0] == 'exito':
                    _, grafo, metricas, dir_salida = resultado
                    self.actualizar_ui_con_resultados(grafo, metricas, dir_salida)
                    self.progreso.stop()
                    self.boton_generar.config(state='normal')
                    self.combo_dataset.config(state='readonly')
                    self.combo_ciudad.config(state='readonly')
                    self.boton_cancelar.config(state='disabled')
                    self.label_estado.config(text="Análisis completado")
                    messagebox.showinfo("Éxito", f"Análisis completado\\nResultados guardados en:\\n{dir_salida}")
                
                elif resultado[0] == 'error':
                    self.progreso.stop()
                    self.boton_generar.config(state='normal')
                    self.combo_dataset.config(state='readonly')
                    self.combo_ciudad.config(state='readonly')
                    self.boton_cancelar.config(state='disabled')
                    self.label_estado.config(text="Error en el análisis")
                    messagebox.showerror("Error", f"Error al procesar la red:\\n{resultado[1]}")
                
                elif resultado[0] == 'cancelado':
                    self.limpiar_paneles()  # Limpiar paneles al cancelar
                    self.progreso.stop()
                    self.boton_generar.config(state='normal')
                    self.combo_dataset.config(state='readonly')
                    self.combo_ciudad.config(state='readonly')
                    self.boton_cancelar.config(state='disabled')
                    self.label_estado.config(text="Listo")
                    messagebox.showinfo("Cancelado", resultado[1])
        
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
        
        # 1. Actualizar mapa
        self.actualizar_mapa(grafo, dir_salida)
        
        # 2. Actualizar tab de info
        self.actualizar_tab_info(metricas)
        
        # 3. Actualizar tab de robustez
        self.actualizar_tab_robustez(metricas)
        
        # 4. Actualizar tab de componentes
        self.actualizar_tab_componentes(grafo)
        
        # 5. Actualizar info de directorio
        self.label_directorio.config(text=f"Resultados guardados en:\\n{dir_salida}")
    
    def actualizar_mapa(self, grafo, dir_salida):
        """Actualiza el panel de mapa con visualización interactiva."""
        # Limpiar canvas anterior
        for widget in self.frame_canvas_mapa.winfo_children():
            widget.destroy()
        
        # Verificar si hay datos geográficos
        tiene_coords = False
        for nodo in grafo.nodos():
            attrs = grafo.atributos_nodos.get(nodo, {})
            if 'lat' in attrs and 'lon' in attrs:
                tiene_coords = True
                break
        
        if tiene_coords:
            # Crear figura interactiva de matplotlib
            fig, ax = plt.subplots(figsize=(8, 7))
            
            # Extraer coordenadas
            lats = []
            lons = []
            for nodo in grafo.nodos():
                attrs = grafo.atributos_nodos.get(nodo, {})
                if 'lat' in attrs and 'lon' in attrs:
                    lats.append(attrs['lat'])
                    lons.append(attrs['lon'])
            
            # Plotear líneas (conexiones)
            for u, v in grafo.aristas():
                u_attrs = grafo.atributos_nodos.get(u, {})
                v_attrs = grafo.atributos_nodos.get(v, {})
                if 'lat' in u_attrs and 'lat' in v_attrs and 'lon' in u_attrs and 'lon' in v_attrs:
                    ax.plot(
                        [u_attrs['lon'], v_attrs['lon']],
                        [u_attrs['lat'], v_attrs['lat']],
                        'b-', alpha=0.3, linewidth=0.8, zorder=1
                    )
            
            # Plotear estaciones
            ax.scatter(lons, lats, c='red', s=60, alpha=0.7, zorder=2, 
                      edgecolors='darkred', linewidths=0.5)
            
            ax.set_xlabel('Longitud', fontsize=11)
            ax.set_ylabel('Latitud', fontsize=11)
            ax.set_title(f'Mapa de Red: {grafo.nombre}', fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            
            # Ajustar límites con margen
            if lats and lons:
                lat_margin = (max(lats) - min(lats)) * 0.1 or 0.01
                lon_margin = (max(lons) - min(lons)) * 0.1 or 0.01
                ax.set_xlim(min(lons) - lon_margin, max(lons) + lon_margin)
                ax.set_ylim(min(lats) - lat_margin, max(lats) + lat_margin)
            
            plt.tight_layout()
            
            # Embeber en tkinter con barra de herramientas
            canvas = FigureCanvasTkAgg(fig, master=self.frame_canvas_mapa)
            canvas.draw()
            
            # Agregar barra de herramientas (zoom, pan, etc.)
            toolbar = NavigationToolbar2Tk(canvas, self.frame_canvas_mapa)
            toolbar.update()
            
            # Empaquetar widgets
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            
        else:
            # No hay datos geográficos
            label = ttk.Label(self.frame_canvas_mapa, 
                            text=f"No hay datos geográficos para\n{grafo.nombre}",
                            font=("Arial", 12))
            label.pack(expand=True)
    
    def actualizar_tab_info(self, metricas):
        """Actualiza el tab de información general."""
        # Limpiar tree
        for item in self.tree_info.get_children():
            self.tree_info.delete(item)
        
        # Agregar métricas
        for clave, valor in metricas.items():
            if isinstance(valor, float):
                valor_str = f"{valor:.4f}"
            else:
                valor_str = str(valor)
            self.tree_info.insert("", tk.END, values=(clave, valor_str))
    
    
    def actualizar_tab_robustez(self, metricas):
        """Actualiza el tab de robustez con gráficos."""
        # Limpiar canvas anterior
        for widget in self.frame_canvas_robustez.winfo_children():
            widget.destroy()
        
        # Crear figura con subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))
        fig.suptitle('Métricas de Robustez', fontsize=16, fontweight='bold')
        
        # 1. Indicadores teóricos (barras)
        if 'r_T' in metricas and 'C_G' in metricas:
            indicadores = ['r_T', 'C_G']
            valores = [metricas['r_T'], metricas['C_G']]
            colores = ['#2ecc71', '#3498db']
            
            bars = ax1.bar(indicadores, valores, color=colores, alpha=0.7, edgecolor='black')
            ax1.set_ylabel('Valor', fontsize=10)
            ax1.set_title('Indicadores Teóricos', fontsize=12, fontweight='bold')
            ax1.set_ylim(0, max(valores) * 1.2 if valores else 1)
            ax1.grid(axis='y', alpha=0.3)
            
            # Agregar valores sobre las barras
            for bar, val in zip(bars, valores):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.4f}',
                        ha='center', va='bottom', fontsize=9)
        else:
            ax1.text(0.5, 0.5, 'No hay datos\ndisponibles', 
                    ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Indicadores Teóricos', fontsize=12, fontweight='bold')
        
        # 2. Robustez por remoción (barras comparativas)
        if 'robustez_grado_20pct' in metricas and 'robustez_aleatorio_20pct' in metricas:
            estrategias = ['Por Grado\n(20%)', 'Aleatoria\n(20%)']
            valores_rob = [metricas['robustez_grado_20pct'], metricas['robustez_aleatorio_20pct']]
            colores_rob = ['#e74c3c', '#f39c12']
            
            bars = ax2.bar(estrategias, valores_rob, color=colores_rob, alpha=0.7, edgecolor='black')
            ax2.set_ylabel('Fracción de nodos\nen componente gigante', fontsize=9)
            ax2.set_title('Robustez por Remoción', fontsize=12, fontweight='bold')
            ax2.set_ylim(0, 1.1)
            ax2.grid(axis='y', alpha=0.3)
            
            # Agregar valores sobre las barras
            for bar, val in zip(bars, valores_rob):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.3f}',
                        ha='center', va='bottom', fontsize=9)
        else:
            ax2.text(0.5, 0.5, 'No hay datos\ndisponibles', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Robustez por Remoción', fontsize=12, fontweight='bold')
        
        # 3. Explicación de r_T
        ax3.axis('off')
        texto_rT = [
            "Indicador r_T (Robustez Topológica):",
            "",
            "• Fórmula: r_T = (L - N + 1) / N",
            "  donde L = aristas, N = nodos",
            "",
            "• Mide la densidad de ciclos",
            "• Valores altos → más rutas alternativas",
            "• Valores bajos → red tipo árbol",
        ]
        if 'r_T' in metricas:
            texto_rT.append("")
            texto_rT.append(f"Valor actual: {metricas['r_T']:.4f}")
        
        ax3.text(0.05, 0.95, '\n'.join(texto_rT), 
                transform=ax3.transAxes, fontsize=9,
                verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        # 4. Explicación de C_G
        ax4.axis('off')
        texto_CG = [
            "Conductancia Efectiva C_G:",
            "",
            "• Basada en resistencia efectiva",
            "• Usa valores propios del Laplaciano",
            "",
            "• C_G ≈ 1: red muy conectada",
            "• C_G ≈ 0: conectividad pobre",
            "• Mide eficiencia de flujo",
        ]
        if 'C_G' in metricas:
            texto_CG.append("")
            texto_CG.append(f"Valor actual: {metricas['C_G']:.4f}")
        
        ax4.text(0.05, 0.95, '\n'.join(texto_CG), 
                transform=ax4.transAxes, fontsize=9,
                verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        plt.tight_layout()
        
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
        
        archivo = filedialog.asksaveasfilename(
            defaultextension=".html",
            filetypes=[("HTML", "*.html"), ("Todos", "*.*")]
        )
        
        if archivo:
            visualizacion._generar_reporte_individual(
                self.grafo_actual,
                self.metricas_actuales,
                Path(archivo)
            )
            messagebox.showinfo("Éxito", f"Reporte generado en:\\n{archivo}")
    
    def ejecutar(self):
        """Ejecuta el loop principal de la aplicación."""
        self.ventana.mainloop()


def main():
    """Función principal para ejecutar la GUI."""
    app = VentanaPrincipal()
    app.ejecutar()


if __name__ == "__main__":
    main()
