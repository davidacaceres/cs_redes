import tkinter as tk
from tkinter import ttk
import networkx as nx
import math

class GrafoInteractivo(tk.Canvas):
    def __init__(self, master, **kwargs):
        super().__init__(master, bg='white', **kwargs)
        
        self.grafo_original = None
        self.G_simplificado = None
        self.posiciones = {} # {nodo_id: (x, y)}
        self.radio_nodo = 15
        
        # Mapeos para interactividad
        self.item_a_nodo = {} # {canvas_item_id: nodo_id}
        self.nodo_a_items = {} # {nodo_id: {'oval': id, 'text': id}}
        self.nodo_a_aristas = {} # {nodo_id: [line_id, ...]}
        self.arista_a_nodos = {} # {line_id: (u, v)}
        
        self.nodo_seleccionado = None
        self.drag_data = {"x": 0, "y": 0}
        self.callback_doble_clic = None
        self.pos_geo_cache = {} # Cache de coordenadas geográficas {nodo_id: (lon, lat)}
        
        # Bindings de Nodos
        self.tag_bind("nodo", "<Button-1>", self._on_click_nodo)
        self.tag_bind("nodo", "<B1-Motion>", self._on_drag_nodo)
        self.tag_bind("nodo", "<ButtonRelease-1>", self._on_release_nodo)
        self.tag_bind("nodo", "<Double-Button-1>", self._on_double_click_nodo)
        
        self.tag_bind("texto", "<Button-1>", self._on_click_nodo)
        self.tag_bind("texto", "<B1-Motion>", self._on_drag_nodo)
        self.tag_bind("texto", "<Double-Button-1>", self._on_double_click_nodo)
        
        # Bindings de Pan y Zoom (Canvas general)
        self.bind("<ButtonPress-3>", self._on_pan_start)
        self.bind("<B3-Motion>", self._on_pan_move)
        # Windows/MacOS
        self.bind("<MouseWheel>", self._on_zoom)
        # Linux
        self.bind("<Button-4>", self._on_zoom)
        self.bind("<Button-5>", self._on_zoom)

    def set_callback_doble_clic(self, callback):
        """Establece la función a llamar al hacer doble clic en un nodo.
        El callback recibirá (lat, lon) como argumentos."""
        self.callback_doble_clic = callback

    def dibujar_grafo(self, grafo_simple):
        """Dibuja el grafo simplificado en el canvas."""
        self.delete("all")
        self.grafo_original = grafo_simple
        self.pos_geo_cache = {}
        
        # 1. Simplificación (Lógica idéntica a visualizacion.py)
        G = nx.Graph()
        G.add_nodes_from(grafo_simple.nodos())
        
        # Cargar aristas con sus atributos (color, etc.)
        for u, v in grafo_simple.aristas():
            # Recuperar atributos usando clave ordenada
            clave = tuple(sorted((u, v)))
            attrs = getattr(grafo_simple, 'atributos_aristas', {}).get(clave, {})
            G.add_edge(u, v, **attrs)
        
        # Extraer posiciones geográficas originales antes de simplificar
        pos_geo = {}
        nodos_sin_pos = []
        for n in G.nodes():
            attrs = grafo_simple.atributos_nodos.get(n, {})
            if 'lon' in attrs and 'lat' in attrs:
                pos_geo[n] = (attrs['lon'], attrs['lat'])
                self.pos_geo_cache[n] = (attrs['lon'], attrs['lat'])
            else:
                nodos_sin_pos.append(n)
        
        # Simplificar nodos de grado 2
        cambios = True
        while cambios:
            cambios = False
            nodos_grado_2 = [n for n in G.nodes() if G.degree(n) == 2]
            for n in nodos_grado_2:
                if G.has_node(n):
                    vecinos = list(G.neighbors(n))
                    if len(vecinos) == 2:
                        u, v = vecinos
                        # Heredar atributos de la arista (priorizar color de u-n o n-v)
                        # Asumimos que si es una línea continua, comparten color
                        attr_u = G.get_edge_data(u, n)
                        attr_v = G.get_edge_data(n, v)
                        
                        # Combinar atributos (simple: tomar de u)
                        new_attr = attr_u.copy() if attr_u else {}
                        
                        if not G.has_edge(u, v):
                            G.add_edge(u, v, **new_attr)
                        G.remove_node(n)
                        cambios = True
        
        # Renumerar nodos (1..N)
        mapping = {node: i for i, node in enumerate(G.nodes(), 1)}
        G = nx.relabel_nodes(G, mapping)
        
        # Actualizar claves de pos_geo y cache
        # Nota: pos_geo_cache debe mantener los IDs originales o mapeados?
        # El grafo simplificado usa IDs nuevos (1..N). Cuando el usuario hace clic,
        # obtenemos el ID del nodo simplificado. Necesitamos saber qué coordenada corresponde.
        # Al renumerar, 'mapping' nos dice old_id -> new_id.
        # Pero pos_geo estaba indexado por old_id.
        # Necesitamos un pos_geo indexado por new_id para el dibujo, y también para el callback.
        
        if pos_geo:
            # Crear nuevo diccionario con claves mapeadas
            new_pos_geo = {}
            for old_id, coords in pos_geo.items():
                if old_id in mapping:
                    new_id = mapping[old_id]
                    new_pos_geo[new_id] = coords
            
            pos_geo = new_pos_geo
            self.pos_geo_cache = new_pos_geo # Actualizar cache con IDs simplificados
            
        self.G_simplificado = G
        
        # 2. Calcular Posiciones de Pantalla
        width = self.winfo_width() or 600
        height = self.winfo_height() or 400
        padding = 50
        
        if pos_geo:
            # Normalizar coordenadas geo a pantalla
            lons = [p[0] for p in pos_geo.values()]
            lats = [p[1] for p in pos_geo.values()]
            
            if lons and lats:
                min_lon, max_lon = min(lons), max(lons)
                min_lat, max_lat = min(lats), max(lats)
                
                range_lon = max_lon - min_lon or 1
                range_lat = max_lat - min_lat or 1
                
                # Mantener aspecto
                scale_x = (width - 2 * padding) / range_lon
                scale_y = (height - 2 * padding) / range_lat
                scale = min(scale_x, scale_y)
                
                # Centrar
                center_x_screen = width / 2
                center_y_screen = height / 2
                center_lon = (min_lon + max_lon) / 2
                center_lat = (min_lat + max_lat) / 2
                
                for n in G.nodes():
                    if n in pos_geo:
                        lon, lat = pos_geo[n]
                        # Y invertido en pantalla (latitud crece hacia arriba, Y pantalla hacia abajo)
                        x = center_x_screen + (lon - center_lon) * scale
                        y = center_y_screen - (lat - center_lat) * scale
                        self.posiciones[n] = (x, y)
                    else:
                        # Fallback para nodos sin geo (raro si simplificamos)
                        self.posiciones[n] = (width/2, height/2)
        else:
            # Layout automático si no hay geo
            pos_spring = nx.spring_layout(G, center=(width/2, height/2), scale=min(width, height)/2 - padding)
            self.posiciones = pos_spring

        # 3. Dibujar Aristas
        self.nodo_a_aristas = {n: [] for n in G.nodes()}
        
        # Paleta de colores (copiada de hmi/mapa.py para consistencia)
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
        color_default = "#808080"  # Gris
        
        for u, v, data in G.edges(data=True):
            x1, y1 = self.posiciones[u]
            x2, y2 = self.posiciones[v]
            
            # Lógica de inferencia de color
            color_arista = data.get('color')
            
            if not color_arista:
                # Intentar inferir desde route_I_counts (formato Metro51)
                route_counts = data.get('route_I_counts', {})
                if route_counts:
                    # Usar la línea con más vehículos
                    try:
                        linea_principal = max(route_counts.items(), key=lambda x: x[1])[0]
                        if str(linea_principal) in colores_lineas:
                            color_arista = colores_lineas[str(linea_principal)]
                        else:
                            # Si no se reconoce la línea, usar gris (como en el mapa)
                            color_arista = color_default
                    except (ValueError, IndexError):
                        color_arista = color_default
                else:
                    color_arista = color_default
            
            line_id = self.create_line(x1, y1, x2, y2, fill=color_arista, width=3, tags="arista")
            self.nodo_a_aristas[u].append(line_id)
            self.nodo_a_aristas[v].append(line_id)
            self.arista_a_nodos[line_id] = (u, v)
            
        # 4. Dibujar Nodos
        for n in G.nodes():
            x, y = self.posiciones[n]
            r = self.radio_nodo
            
            # Color
            grado = G.degree(n)
            color = "white" if grado == 1 else "#FFC107"
            
            oval_id = self.create_oval(x-r, y-r, x+r, y+r, fill=color, outline="black", width=2, tags=("nodo", f"nodo_{n}"))
            text_id = self.create_text(x, y, text=str(n), fill="black", font=("Arial", 8, "bold"), tags=("texto", f"texto_{n}"))
            
            self.item_a_nodo[oval_id] = n
            self.item_a_nodo[text_id] = n
            self.nodo_a_items[n] = {'oval': oval_id, 'text': text_id}

    def _get_nodo_center(self, nodo_id):
        """Calcula el centro actual de un nodo en el canvas."""
        oval_id = self.nodo_a_items[nodo_id]['oval']
        coords = self.coords(oval_id)
        if coords:
            x1, y1, x2, y2 = coords
            return (x1 + x2) / 2, (y1 + y2) / 2
        return 0, 0

    def _on_click_nodo(self, event):
        """Inicia el arrastre."""
        # Convertir coordenadas de pantalla a canvas (necesario si hubo pan/zoom)
        canvas_x = self.canvasx(event.x)
        canvas_y = self.canvasy(event.y)
        
        item = self.find_closest(canvas_x, canvas_y)[0]
        nodo_id = self.item_a_nodo.get(item)
        
        if nodo_id:
            self.nodo_seleccionado = nodo_id
            self.drag_data["x"] = canvas_x
            self.drag_data["y"] = canvas_y
            
            oval_id = self.nodo_a_items[nodo_id]['oval']
            self.itemconfig(oval_id, outline="red")

    def _on_drag_nodo(self, event):
        """Mueve el nodo y sus aristas."""
        if self.nodo_seleccionado:
            canvas_x = self.canvasx(event.x)
            canvas_y = self.canvasy(event.y)
            
            dx = canvas_x - self.drag_data["x"]
            dy = canvas_y - self.drag_data["y"]
            
            # Mover oval y texto
            oval_id = self.nodo_a_items[self.nodo_seleccionado]['oval']
            text_id = self.nodo_a_items[self.nodo_seleccionado]['text']
            self.move(oval_id, dx, dy)
            self.move(text_id, dx, dy)
            
            # Actualizar aristas conectadas usando coordenadas reales
            # (Ya no dependemos de self.posiciones que puede desincronizarse con zoom)
            cx, cy = self._get_nodo_center(self.nodo_seleccionado)
            
            for line_id in self.nodo_a_aristas[self.nodo_seleccionado]:
                u, v = self.arista_a_nodos[line_id]
                # Determinar el otro extremo
                otro_nodo = v if u == self.nodo_seleccionado else u
                ox, oy = self._get_nodo_center(otro_nodo)
                
                self.coords(line_id, cx, cy, ox, oy)
            
            self.drag_data["x"] = canvas_x
            self.drag_data["y"] = canvas_y

    def _on_release_nodo(self, event):
        """Finaliza el arrastre."""
        if self.nodo_seleccionado:
            oval_id = self.nodo_a_items[self.nodo_seleccionado]['oval']
            self.itemconfig(oval_id, outline="black")
            self.nodo_seleccionado = None

    def _on_pan_start(self, event):
        """Inicia el paneo."""
        self.scan_mark(event.x, event.y)

    def _on_pan_move(self, event):
        """Mueve la vista."""
        self.scan_dragto(event.x, event.y, gain=1)

    def _on_zoom(self, event):
        """Realiza zoom in/out."""
        scale = 1.0
        # Windows/MacOS: event.delta
        if event.delta:
            if event.delta > 0:
                scale = 1.1
            else:
                scale = 0.9
        # Linux: num (4 scroll up, 5 scroll down)
        elif event.num == 4:
            scale = 1.1
        elif event.num == 5:
            scale = 0.9
            
        x = self.canvasx(event.x)
        y = self.canvasy(event.y)
        
        self.scale("all", x, y, scale, scale)

    def _on_double_click_nodo(self, event):
        """Maneja el doble clic en un nodo para centrar el mapa."""
        # Convertir coordenadas de pantalla a canvas
        canvas_x = self.canvasx(event.x)
        canvas_y = self.canvasy(event.y)
        
        # Encontrar item más cercano
        item = self.find_closest(canvas_x, canvas_y)[0]
        nodo_id = self.item_a_nodo.get(item)
        
        if nodo_id and self.callback_doble_clic:
            # Buscar coordenadas geográficas
            coords = self.pos_geo_cache.get(nodo_id)
            if coords:
                lon, lat = coords
                # Llamar al callback con (lat, lon)
                self.callback_doble_clic(lat, lon)
                print(f"[INFO] Doble clic en nodo {nodo_id}. Centrando mapa en ({lat}, {lon})")
            else:
                print(f"[WARN] Nodo {nodo_id} no tiene coordenadas geográficas asociadas.")
