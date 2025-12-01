"""
Módulo de Mapa Interactivo
---------------------------

Proporciona un widget de mapa interactivo usando tkintermapview
con overlays personalizados para visualizar redes de transporte.
"""

import tkinter as tk
from tkinter import ttk
import tkintermapview
from tkintermapview.utility_functions import decimal_to_osm


class FastNetworkOverlay:
    """Overlay optimizado para renderizar aristas de red con colores por línea."""
    
    def __init__(self, map_widget, edges_list, color="blue", width=1):
        """
        Inicializa el overlay de aristas.
        
        Parámetros
        ----------
        map_widget : TkinterMapView
            Widget del mapa
        edges_list : list
            Lista de aristas. Cada elemento puede ser:
            - Tupla de 2 coordenadas: ((lat1, lon1), (lat2, lon2))
            - Tupla de 3 elementos: ((lat1, lon1), (lat2, lon2), color)
        color : str
            Color por defecto si no se especifica en la arista
        width : int
            Ancho de línea
        """
        self.map_widget = map_widget
        self.edges_list = edges_list
        self.default_color = color
        self.width = width
        self.tag_name = f"network_edges_{id(self)}"
        self.deleted = False
        self.last_upper_left_tile_pos = None
        
        # Tooltip
        self.tooltip_label = None
        self._setup_tooltip()

    def _setup_tooltip(self):
        """Configura el label del tooltip."""
        self.tooltip_label = tk.Label(
            self.map_widget, 
            text="", 
            bg="#ffffe0", 
            relief="solid", 
            borderwidth=1,
            font=("Arial", 8),
            justify=tk.LEFT
        )

    def _on_enter(self, event):
        """Maneja el evento de entrada del mouse sobre una arista."""
        # Encontrar el ítem del canvas bajo el mouse
        item = self.map_widget.canvas.find_withtag("current")
        if not item:
            return
            
        tags = self.map_widget.canvas.gettags(item)
        edge_index = -1
        
        # Buscar tag de índice
        for tag in tags:
            if tag.startswith("edge_idx_"):
                try:
                    edge_index = int(tag.split("_")[-1])
                    break
                except ValueError:
                    pass
        
        if edge_index >= 0 and edge_index < len(self.edges_list):
            edge_data = self.edges_list[edge_index]
            # Extraer atributos si existen (4to elemento)
            if len(edge_data) >= 4:
                attrs = edge_data[3]
                if attrs:
                    # Formatear texto del tooltip
                    text = "Información de Arista:\n"
                    for k, v in attrs.items():
                        # Filtrar atributos internos o muy largos si es necesario
                        if k not in ['geometry', 'osmid']: 
                            text += f"{k}: {v}\n"
                    
                    self.tooltip_label.config(text=text.strip())
                    
                    # Posicionar tooltip cerca del mouse
                    x, y = event.x_root - self.map_widget.winfo_rootx() + 15, event.y_root - self.map_widget.winfo_rooty() + 15
                    
                    # Ajustar si se sale de la pantalla (básico)
                    if x + self.tooltip_label.winfo_reqwidth() > self.map_widget.winfo_width():
                        x -= self.tooltip_label.winfo_reqwidth() + 20
                        
                    self.tooltip_label.place(x=x, y=y)
                    self.tooltip_label.lift()

    def _on_leave(self, event):
        """Maneja el evento de salida del mouse."""
        if self.tooltip_label:
            self.tooltip_label.place_forget()
        
        # Tooltip
        self.tooltip_label = None
        self._setup_tooltip()

    def _setup_tooltip(self):
        """Configura el label del tooltip."""
        self.tooltip_label = tk.Label(
            self.map_widget, 
            text="", 
            bg="#ffffe0", 
            relief="solid", 
            borderwidth=1,
            font=("Arial", 8),
            justify=tk.LEFT
        )

    def _on_enter(self, event):
        """Maneja el evento de entrada del mouse sobre una arista."""
        # Encontrar el ítem del canvas bajo el mouse
        item = self.map_widget.canvas.find_withtag("current")
        if not item:
            return
            
        tags = self.map_widget.canvas.gettags(item)
        edge_index = -1
        
        # Buscar tag de índice
        for tag in tags:
            if tag.startswith("edge_idx_"):
                try:
                    edge_index = int(tag.split("_")[-1])
                    break
                except ValueError:
                    pass
        
        if edge_index >= 0 and edge_index < len(self.edges_list):
            edge_data = self.edges_list[edge_index]
            # Extraer atributos si existen (4to elemento)
            if len(edge_data) >= 4:
                attrs = edge_data[3]
                if attrs:
                    # Formatear texto del tooltip
                    text = ""
                    for k, v in attrs.items():
                        # Filtrar atributos internos o muy largos si es necesario
                        if k == 'duration_avg':
                            text += f"Tiempo promedio: {v} [seg]\n"
                        elif k == 'd':
                            text += f"Distancia: {v} mtrs.\n"
                        elif k not in ['geometry', 'osmid','shape_id','n_vehicles','route_I_counts','direction_id','headsign']: 
                            text += f"{k}: {v}\n"
                    
                    self.tooltip_label.config(text=text.strip())
                    
                    # Posicionar tooltip cerca del mouse
                    x, y = event.x_root - self.map_widget.winfo_rootx() + 15, event.y_root - self.map_widget.winfo_rooty() + 15
                    
                    # Ajustar si se sale de la pantalla (básico)
                    if x + self.tooltip_label.winfo_reqwidth() > self.map_widget.winfo_width():
                        x -= self.tooltip_label.winfo_reqwidth() + 20
                        
                    self.tooltip_label.place(x=x, y=y)
                    self.tooltip_label.lift()

    def _on_leave(self, event):
        """Maneja el evento de salida del mouse."""
        if self.tooltip_label:
            self.tooltip_label.place_forget()
        
    def delete(self):
        self.deleted = True
        self.map_widget.canvas.delete(self.tag_name)

    def set_position_list(self, position_list):
        pass

    def draw(self, move=False):
        if self.deleted:
            return
        
        # Obtener dimensiones de tile
        if hasattr(self.map_widget, 'canvas_tile_array') and self.map_widget.canvas_tile_array:
            try:
                first_tile = self.map_widget.canvas_tile_array[0][0]
                if first_tile:
                    widget_tile_width = first_tile.widget_tile_width
                    widget_tile_height = first_tile.widget_tile_height
                else:
                    return
            except (IndexError, AttributeError):
                return
        else:
            return

        if move and self.last_upper_left_tile_pos is not None:
            scale_x = self.map_widget.width / widget_tile_width
            scale_y = self.map_widget.height / widget_tile_height
            
            x_move = (self.last_upper_left_tile_pos[0] - self.map_widget.upper_left_tile_pos[0]) * scale_x
            y_move = (self.last_upper_left_tile_pos[1] - self.map_widget.upper_left_tile_pos[1]) * scale_y
            
            self.map_widget.canvas.move(self.tag_name, x_move, y_move)
        else:
            self.map_widget.canvas.delete(self.tag_name)
            
            zoom = round(self.map_widget.zoom)
            upper_left = self.map_widget.upper_left_tile_pos
            
            scale_x = self.map_widget.width / widget_tile_width
            scale_y = self.map_widget.height / widget_tile_height
            
            for i, edge_data in enumerate(self.edges_list):
                # Soportar formato con o sin color y atributos
                # Formatos: (p1, p2), (p1, p2, color), (p1, p2, color, attrs)
                edge_color = self.default_color
                
                if len(edge_data) >= 3:
                    p1 = edge_data[0]
                    p2 = edge_data[1]
                    edge_color = edge_data[2]
                else:
                    p1, p2 = edge_data
                
                tile_pos1 = decimal_to_osm(p1[0], p1[1], zoom)
                x1 = (tile_pos1[0] - upper_left[0]) * scale_x
                y1 = (tile_pos1[1] - upper_left[1]) * scale_y
                
                tile_pos2 = decimal_to_osm(p2[0], p2[1], zoom)
                x2 = (tile_pos2[0] - upper_left[0]) * scale_x
                y2 = (tile_pos2[1] - upper_left[1]) * scale_y
                
                # Culling simple
                if (x1 < 0 and x2 < 0) or (x1 > self.map_widget.width and x2 > self.map_widget.width) or \
                   (y1 < 0 and y2 < 0) or (y1 > self.map_widget.height and y2 > self.map_widget.height):
                    continue
                
                # Crear línea con tags para identificación
                line_id = self.map_widget.canvas.create_line(x1, y1, x2, y2, fill=edge_color, width=self.width, 
                                                   capstyle="round", joinstyle="round",
                                                   tag=(self.tag_name, "path", f"edge_idx_{i}"))
                
                # Bindings para tooltip (solo si hay atributos)
                if len(edge_data) >= 4 and edge_data[3]:
                    self.map_widget.canvas.tag_bind(line_id, "<Enter>", self._on_enter)
                    self.map_widget.canvas.tag_bind(line_id, "<Leave>", self._on_leave)
        
        self.map_widget.canvas.tag_raise(self.tag_name)
        self.last_upper_left_tile_pos = self.map_widget.upper_left_tile_pos
            



class FastNodeOverlay:
    """Overlay optimizado para renderizar nodos de red."""
    
    def __init__(self, map_widget, nodes_list, color="red", radius=3):
        self.map_widget = map_widget
        self.nodes_list = nodes_list
        self.color = color
        self.radius = radius
        self.tag_name = f"network_nodes_{id(self)}"
        self.deleted = False
        self.last_upper_left_tile_pos = None
        
        # Tooltip
        self.tooltip_label = None
        self._setup_tooltip()
        
    def _setup_tooltip(self):
        """Configura el label del tooltip."""
        self.tooltip_label = tk.Label(
            self.map_widget, 
            text="", 
            bg="#ffffe0", 
            relief="solid", 
            borderwidth=1,
            font=("Arial", 8),
            justify=tk.LEFT
        )

    def _on_enter(self, event):
        """Maneja el evento de entrada del mouse sobre un nodo."""
        item = self.map_widget.canvas.find_withtag("current")
        if not item:
            return
            
        tags = self.map_widget.canvas.gettags(item)
        node_index = -1
        
        for tag in tags:
            if tag.startswith("node_idx_"):
                try:
                    node_index = int(tag.split("_")[-1])
                    break
                except ValueError:
                    pass
        
        if node_index >= 0 and node_index < len(self.nodes_list):
            node_data = self.nodes_list[node_index]
            # Extraer atributos si existen (3er elemento)
            if len(node_data) >= 3:
                attrs = node_data[2]
                if attrs:
                    text = ""
                    for k, v in attrs.items():
                        if k == 'name':
                            text += f"Estación: {v}\n"
                        elif k == 'lat':
                            text += f"Latitud: {v}\n"
                        elif k == 'lon':
                            text += f"Longitud: {v}\n"
                        if k not in ['geometry', 'osmid', 'y', 'x', 'street_count', 'highway','original_ids','lat','lon','name']: 
                            text += f"{k}: {v}\n"
                    
                    self.tooltip_label.config(text=text.strip())
                    
                    x, y = event.x_root - self.map_widget.winfo_rootx() + 15, event.y_root - self.map_widget.winfo_rooty() + 15
                    
                    if x + self.tooltip_label.winfo_reqwidth() > self.map_widget.winfo_width():
                        x -= self.tooltip_label.winfo_reqwidth() + 20
                        
                    self.tooltip_label.place(x=x, y=y)
                    self.tooltip_label.lift()

    def _on_leave(self, event):
        """Maneja el evento de salida del mouse."""
        if self.tooltip_label:
            self.tooltip_label.place_forget()

    def delete(self):
        self.deleted = True
        self.map_widget.canvas.delete(self.tag_name)
        if self.tooltip_label:
            self.tooltip_label.destroy()
            self.tooltip_label = None

    def set_position_list(self, position_list):
        pass

    def draw(self, move=False):
        if self.deleted:
            return
        
        # Obtener dimensiones de tile
        if hasattr(self.map_widget, 'canvas_tile_array') and self.map_widget.canvas_tile_array:
            try:
                first_tile = self.map_widget.canvas_tile_array[0][0]
                if first_tile:
                    widget_tile_width = first_tile.widget_tile_width
                    widget_tile_height = first_tile.widget_tile_height
                else:
                    return
            except (IndexError, AttributeError):
                return
        else:
            return

        if move and self.last_upper_left_tile_pos is not None:
            scale_x = self.map_widget.width / widget_tile_width
            scale_y = self.map_widget.height / widget_tile_height
            
            x_move = (self.last_upper_left_tile_pos[0] - self.map_widget.upper_left_tile_pos[0]) * scale_x
            y_move = (self.last_upper_left_tile_pos[1] - self.map_widget.upper_left_tile_pos[1]) * scale_y
            
            self.map_widget.canvas.move(self.tag_name, x_move, y_move)
        else:
            self.map_widget.canvas.delete(self.tag_name)
            
            zoom = round(self.map_widget.zoom)
            upper_left = self.map_widget.upper_left_tile_pos
            
            scale_x = self.map_widget.width / widget_tile_width
            scale_y = self.map_widget.height / widget_tile_height
            
            r = self.radius
            for i, node_data in enumerate(self.nodes_list):
                # Soportar formato (lat, lon) o (lat, lon, attrs)
                if len(node_data) >= 2:
                    lat, lon = node_data[0], node_data[1]
                else:
                    continue

                tile_pos = decimal_to_osm(lat, lon, zoom)
                x = (tile_pos[0] - upper_left[0]) * scale_x
                y = (tile_pos[1] - upper_left[1]) * scale_y
                
                # Culling simple
                if (x < -r or x > self.map_widget.width + r) or \
                   (y < -r or y > self.map_widget.height + r):
                    continue
                
                # Crear marcador con tags
                marker_id = self.map_widget.canvas.create_oval(x-r, y-r, x+r, y+r, fill=self.color, 
                                                   outline="darkred", width=1, 
                                                   tag=(self.tag_name, "marker", f"node_idx_{i}"))
                
                # Bindings para tooltip
                if len(node_data) >= 3 and node_data[2]:
                    self.map_widget.canvas.tag_bind(marker_id, "<Enter>", self._on_enter)
                    self.map_widget.canvas.tag_bind(marker_id, "<Leave>", self._on_leave)
        
        self.map_widget.canvas.tag_raise(self.tag_name)
        self.last_upper_left_tile_pos = self.map_widget.upper_left_tile_pos


class MapaWidget:
    """Widget de mapa interactivo para visualizar redes de transporte."""
    
    def __init__(self, parent_frame):
        """
        Inicializa el widget del mapa.
        
        Parámetros
        ----------
        parent_frame : tk.Frame
            Frame padre donde se colocará el mapa
        """
        self.parent_frame = parent_frame
        self.map_widget = None
        self.network_overlay = None
        self.nodes_overlay = None
        self.original_bounds = None
        
        # Variables para zoom por área
        self.zoom_rect_id = None
        self.start_x = None
        self.start_y = None
        
    def crear_mapa(self):
        """Crea el widget de mapa si no existe."""
        if self.map_widget is None or not self.map_widget.winfo_exists():
            self.map_widget = tkintermapview.TkinterMapView(self.parent_frame, corner_radius=0)
            self.map_widget.pack(fill="both", expand=True)
            return True
        return False
    
    def limpiar(self):
        """Limpia el mapa y sus overlays."""
        if self.map_widget and self.map_widget.winfo_exists():
            self.map_widget.delete_all_marker()
            self.map_widget.delete_all_path()
            
            # Limpiar overlays
            if self.network_overlay:
                if self.network_overlay in self.map_widget.canvas_path_list:
                    self.map_widget.canvas_path_list.remove(self.network_overlay)
                self.network_overlay.delete()
                self.network_overlay = None
                
            if self.nodes_overlay:
                if self.nodes_overlay in self.map_widget.canvas_path_list:
                    self.map_widget.canvas_path_list.remove(self.nodes_overlay)
                self.nodes_overlay.delete()
                self.nodes_overlay = None
    
    def actualizar_con_grafo(self, grafo):
        """
        Actualiza el mapa con un grafo.
        
        Parámetros
        ----------
        grafo : GrafoSimple
            Grafo con datos geográficos
            
        Retorna
        -------
        bool
            True si se pudo visualizar, False si no hay datos geográficos
        """
        # Verificar si hay datos geográficos
        tiene_coords = False
        for nodo in grafo.nodos():
            attrs = grafo.atributos_nodos.get(nodo, {})
            if 'lat' in attrs and 'lon' in attrs:
                tiene_coords = True
                break
        
        if not tiene_coords:
            return False
        
        # Crear mapa si no existe
        self.crear_mapa()
        
        # Limpiar overlays anteriores
        self.limpiar()
        
        # Recopilar coordenadas
        lats = []
        lons = []
        edges_coords = []
        nodes_coords = []
        
        # Paleta de colores para líneas de metro
        colores_lineas = {
            "1": "#E2231A",  # Rojo
            "2": "#00A651",  # Verde
            "3": "#F7931D",  # Naranja
            "4": "#009FE3",  # Azul claro
            "5": "#FFD100",  # Amarillo
            "6": "#8B5A3C",  # Marrón
            "7": "#EE4B9B",  # Rosa
            "8": "#00A99D",  # Turquesa
            "9": "#9B59B6",  # Púrpura
            "10": "#34495E", # Gris oscuro
        }
        color_default = "#808080"  # Gris para líneas sin ID
        
        # Recopilar aristas con colores
        for u, v in grafo.aristas():
            u_attrs = grafo.atributos_nodos.get(u, {})
            v_attrs = grafo.atributos_nodos.get(v, {})
            
            if 'lat' in u_attrs and 'lon' in u_attrs and 'lat' in v_attrs and 'lon' in v_attrs:
                lats.extend([u_attrs['lat'], v_attrs['lat']])
                lons.extend([u_attrs['lon'], v_attrs['lon']])
                
                # Obtener color según route_I_counts
                edge_attrs = grafo.atributos_aristas.get((u, v), {})
                route_counts = edge_attrs.get('route_I_counts', {})
                
                # Usar la línea con más vehículos
                if route_counts:
                    linea_principal = max(route_counts.items(), key=lambda x: x[1])[0]
                    color = colores_lineas.get(str(linea_principal), color_default)
                else:
                    color = color_default
                
                edges_coords.append([(u_attrs['lat'], u_attrs['lon']), 
                                    (v_attrs['lat'], v_attrs['lon']),
                                    color,
                                    edge_attrs]) # Pasar atributos completos
        
        # Recopilar nodos
        for nodo in grafo.nodos():
            attrs = grafo.atributos_nodos.get(nodo, {})
            if 'lat' in attrs and 'lon' in attrs:
                nodes_coords.append((attrs['lat'], attrs['lon'], attrs))
        
        # Crear overlays
        if edges_coords:
            self.network_overlay = FastNetworkOverlay(self.map_widget, edges_coords, 
                                                      color="blue", width=5)
            self.map_widget.canvas_path_list.append(self.network_overlay)
            print(f"[INFO] Agregadas {len(edges_coords)} aristas con colores por línea")
        
        if nodes_coords:
            self.nodes_overlay = FastNodeOverlay(self.map_widget, nodes_coords, 
                                                 color="red", radius=3)
            self.map_widget.canvas_path_list.append(self.nodes_overlay)
            print(f"[INFO] Agregados {len(nodes_coords)} nodos usando FastNodeOverlay")
        
        # Ajustar vista
        if lats and lons:
            max_lat, min_lat = max(lats), min(lats)
            max_lon, min_lon = max(lons), min(lons)
            self.map_widget.fit_bounding_box((max_lat, min_lon), (min_lat, max_lon))
            self.original_bounds = ((max_lat, min_lon), (min_lat, max_lon))
        
        return True
    
    # def zoom_in(self):
    #     """Acerca el zoom."""
    #     if self.map_widget:
    #         self.map_widget.set_zoom(self.map_widget.zoom + 1)
    
    # def zoom_out(self):
    #     """Aleja el zoom."""
    #     if self.map_widget:
    #         self.map_widget.set_zoom(self.map_widget.zoom - 1)
    
    def centrar(self):
        """Centra el mapa en los bounds originales."""
        if self.map_widget and self.original_bounds:
            self.map_widget.fit_bounding_box(self.original_bounds[0], self.original_bounds[1])
            
    def mostrar_sonar(self, lat, lon, duracion_ms=2000):
        """Muestra una animación de sonar en la ubicación especificada."""
        if not self.map_widget:
            return
            
        sonar = SonarOverlay(self.map_widget, lat, lon, duracion_ms)
        self.map_widget.canvas_path_list.append(sonar)
        sonar.iniciar_animacion()
    
    # def activar_pan(self):
    #     """Activa el modo panorámico (navegación normal)."""
    #     if self.map_widget:
    #         self.map_widget.canvas.bind("<Button-1>", self.map_widget.mouse_click)
    #         self.map_widget.canvas.bind("<B1-Motion>", self.map_widget.mouse_move)
    #         self.map_widget.canvas.bind("<ButtonRelease-1>", self.map_widget.mouse_release)
    #         self.map_widget.canvas.config(cursor="arrow")
            
    #         if self.zoom_rect_id:
    #             self.map_widget.canvas.delete(self.zoom_rect_id)
    #             self.zoom_rect_id = None
    
    def activar_zoom_area(self):
        """Activa el modo zoom por área."""
        if self.map_widget:
            self.map_widget.canvas.bind("<Button-1>", self._start_zoom_rect)
            self.map_widget.canvas.bind("<B1-Motion>", self._drag_zoom_rect)
            self.map_widget.canvas.bind("<ButtonRelease-1>", self._end_zoom_rect)
            self.map_widget.canvas.config(cursor="crosshair")
    
    def _start_zoom_rect(self, event):
        """Inicia el rectángulo de zoom."""
        self.start_x = self.map_widget.canvas.canvasx(event.x)
        self.start_y = self.map_widget.canvas.canvasy(event.y)
        
        if self.zoom_rect_id:
            self.map_widget.canvas.delete(self.zoom_rect_id)
        
        self.zoom_rect_id = self.map_widget.canvas.create_rectangle(
            self.start_x, self.start_y, self.start_x, self.start_y,
            outline="red", width=2, dash=(4, 4)
        )
    
    def _drag_zoom_rect(self, event):
        """Arrastra el rectángulo de zoom."""
        cur_x = self.map_widget.canvas.canvasx(event.x)
        cur_y = self.map_widget.canvas.canvasy(event.y)
        
        if self.zoom_rect_id:
            self.map_widget.canvas.coords(self.zoom_rect_id, self.start_x, self.start_y, cur_x, cur_y)
    
    def _end_zoom_rect(self, event):
        """Finaliza el zoom por área."""
        if not self.start_x or not self.start_y:
            return
        
        end_x = self.map_widget.canvas.canvasx(event.x)
        end_y = self.map_widget.canvas.canvasy(event.y)
        
        if self.zoom_rect_id:
            self.map_widget.canvas.delete(self.zoom_rect_id)
            self.zoom_rect_id = None
        
        # Evitar zoom en clicks simples
        if abs(end_x - self.start_x) < 10 or abs(end_y - self.start_y) < 10:
            return
        
        try:
            lat1, lon1 = self.map_widget.convert_canvas_coords_to_decimal_coords(self.start_x, self.start_y)
            lat2, lon2 = self.map_widget.convert_canvas_coords_to_decimal_coords(end_x, end_y)
            self.map_widget.fit_bounding_box((lat1, lon1), (lat2, lon2))
            self.activar_pan()
        except Exception as e:
            print(f"Error en zoom area: {e}")
    
    def cambiar_proveedor(self, proveedor):
        """
        Cambia el proveedor de tiles del mapa.
        
        Parámetros
        ----------
        proveedor : str
            Nombre del proveedor
        """
        if not self.map_widget:
            return
        
        if proveedor == "OpenStreetMap (Default)":
            self.map_widget.set_tile_server("https://a.tile.openstreetmap.org/{z}/{x}/{y}.png")
            self.map_widget.set_overlay_tile_server(None)
        elif proveedor == "Google Maps Normal":
            self.map_widget.set_tile_server("https://mt0.google.com/vt/lyrs=m&hl=en&x={x}&y={y}&z={z}&s=Ga")
            self.map_widget.set_overlay_tile_server(None)
        elif proveedor == "Google Maps Satélite":
            self.map_widget.set_tile_server("https://mt0.google.com/vt/lyrs=s&hl=en&x={x}&y={y}&z={z}&s=Ga")
            self.map_widget.set_overlay_tile_server(None)
        elif proveedor == "Google Maps Híbrido":
            self.map_widget.set_tile_server("https://mt0.google.com/vt/lyrs=s&hl=en&x={x}&y={y}&z={z}&s=Ga")
            self.map_widget.set_overlay_tile_server("https://mt0.google.com/vt/lyrs=h&hl=en&x={x}&y={y}&z={z}&s=Ga")
        elif proveedor == "OpenTopoMap":
            self.map_widget.set_tile_server("https://a.tile.opentopomap.org/{z}/{x}/{y}.png")
            self.map_widget.set_overlay_tile_server(None)
    
    def destruir(self):
        """Destruye el widget del mapa."""
        if self.map_widget:
            self.map_widget.destroy()
            self.map_widget = None


class SonarOverlay:
    """Overlay animado tipo sonar que se expande y desaparece."""
    
    def __init__(self, map_widget, lat, lon, duration_ms=2000):
        self.map_widget = map_widget
        self.lat = lat
        self.lon = lon
        self.duration_ms = duration_ms
        self.start_time = None
        self.tag_name = f"sonar_{id(self)}"
        self.deleted = False
        self.radius = 0
        self.max_radius = 100 # Pixeles
        self.steps = 50
        self.current_step = 0
        self.interval = duration_ms // self.steps
        
    def delete(self):
        self.deleted = True
        self.map_widget.canvas.delete(self.tag_name)
        
    def set_position_list(self, position_list):
        pass
        
    def draw(self, move=False):
        # Este método es llamado por el mapa cuando se mueve/zoomea.
        if self.deleted:
            return
            
        # Obtener dimensiones de tile (Lógica copiada de FastNetworkOverlay)
        if hasattr(self.map_widget, 'canvas_tile_array') and self.map_widget.canvas_tile_array:
            try:
                first_tile = self.map_widget.canvas_tile_array[0][0]
                if first_tile:
                    widget_tile_width = first_tile.widget_tile_width
                    widget_tile_height = first_tile.widget_tile_height
                else:
                    return
            except (IndexError, AttributeError):
                return
        else:
            return

        # Calcular posición en pantalla manualmente
        zoom = round(self.map_widget.zoom)
        upper_left = self.map_widget.upper_left_tile_pos
        scale_x = self.map_widget.width / widget_tile_width
        scale_y = self.map_widget.height / widget_tile_height
        
        tile_pos = decimal_to_osm(self.lat, self.lon, zoom)
        x = (tile_pos[0] - upper_left[0]) * scale_x
        y = (tile_pos[1] - upper_left[1]) * scale_y
            
        r = self.radius
        
        # Si ya existe, moverlo/actualizarlo
        if self.map_widget.canvas.find_withtag(self.tag_name):
            self.map_widget.canvas.coords(self.tag_name, x-r, y-r, x+r, y+r)
        else:
            # Si no existe (primer draw o borrado por mapa), crearlo
            self.map_widget.canvas.create_oval(x-r, y-r, x+r, y+r, 
                                             outline="#0078D7", width=3, 
                                             tags=self.tag_name)

    def iniciar_animacion(self):
        self.animate_step()
        
    def animate_step(self):
        if self.deleted or self.current_step >= self.steps:
            self.delete()
            if self in self.map_widget.canvas_path_list:
                self.map_widget.canvas_path_list.remove(self)
            return
            
        # Actualizar radio
        progress = self.current_step / self.steps
        self.radius = self.max_radius * progress
        
        # Redibujar (actualizar coords)
        self.draw()
        
        # Efecto de desvanecimiento (simulado reduciendo ancho)
        width = max(1, int(5 * (1 - progress)))
        self.map_widget.canvas.itemconfig(self.tag_name, width=width)
        
        self.current_step += 1
        self.map_widget.after(self.interval, self.animate_step)
