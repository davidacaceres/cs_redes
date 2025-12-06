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
        self.id_mapping = {} # Mapeo de Original -> Display ID
        
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
        self.bind("<Control-MouseWheel>", self._on_zoom)
        
        self.bind("<ButtonPress-1>", self._on_drag_start)
        self.bind("<B1-Motion>", self._on_drag_move)
        
        # Binding para redibujar al cambiar tamaño (Responsive)
        self.bind("<Configure>", self._on_resize)
        
        # Variables de estado para redibujado
        self.world_pos = {} # Coordenadas "mundo" (lon/lat o abstractas)
        self.world_bounds = None # (min_x, max_x, min_y, max_y)
        self.posiciones_screen = {} # Cache de coords pantalla para hit testing

    def set_callback_doble_clic(self, callback):
        """Establece la función a llamar al hacer doble clic en un nodo.
        El callback recibirá (lat, lon) como argumentos."""
        self.callback_doble_clic = callback
        
    def _on_resize(self, event):
        """Redibuja el grafo cuando cambia el tamaño del widget."""
        if self.G_simplificado:
            self._redibujar()

    def dibujar_grafo(self, grafo_simple):
        """Dibuja el grafo simplificado en el canvas."""
        self.delete("all")
        self.grafo_original = grafo_simple
        self.pos_geo_cache = {}
        
        # 1. Simplificación
        G = nx.Graph()
        G.add_nodes_from(grafo_simple.nodos())
        
        for u, v in grafo_simple.aristas():
            clave = tuple(sorted((u, v)))
            attrs = getattr(grafo_simple, 'atributos_aristas', {}).get(clave, {})
            G.add_edge(u, v, **attrs)
        
        pos_geo = {}
        for n in G.nodes():
            attrs = grafo_simple.atributos_nodos.get(n, {})
            if 'lon' in attrs and 'lat' in attrs:
                pos_geo[n] = (attrs['lon'], attrs['lat'])
                self.pos_geo_cache[n] = (attrs['lon'], attrs['lat'])
        
        cambios = True
        while cambios:
            cambios = False
            nodos_grado_2 = [n for n in G.nodes() if G.degree(n) == 2]
            for n in nodos_grado_2:
                if G.has_node(n):
                    vecinos = list(G.neighbors(n))
                    if len(vecinos) == 2:
                        u, v = vecinos
                        attr_u = G.get_edge_data(u, n)
                        new_attr = attr_u.copy() if attr_u else {}
                        
                        if not G.has_edge(u, v):
                            G.add_edge(u, v, **new_attr)
                        G.remove_node(n)
                        cambios = True
        
        mapping = {node: i for i, node in enumerate(G.nodes(), 1)}
        self.id_mapping = mapping
        G = nx.relabel_nodes(G, mapping)
        
        if pos_geo:
            new_pos_geo = {}
            for old_id, coords in pos_geo.items():
                if old_id in mapping:
                    new_id = mapping[old_id]
                    new_pos_geo[new_id] = coords
            pos_geo = new_pos_geo
            self.pos_geo_cache = new_pos_geo
            
        self.G_simplificado = G
        
        # 2. Calcular Coordenadas Mundiales (Abstractas)
        self.world_pos = {}
        min_x, max_x = float('inf'), float('-inf')
        min_y, max_y = float('inf'), float('-inf')
        
        if pos_geo:
            for n in G.nodes():
                if n in pos_geo:
                    lon, lat = pos_geo[n]
                    self.world_pos[n] = (lon, lat)
                    min_x = min(min_x, lon)
                    max_x = max(max_x, lon)
                    min_y = min(min_y, lat)
                    max_y = max(max_y, lat)
                else:
                    self.world_pos[n] = (0, 0)
        else:
            pos_spring = nx.spring_layout(G)
            for n, coords in pos_spring.items():
                x, y = coords[0], coords[1]
                self.world_pos[n] = (x, y)
                min_x = min(min_x, x)
                max_x = max(max_x, x)
                min_y = min(min_y, y)
                max_y = max(max_y, y)
        
        if not self.world_pos:
            self.world_bounds = (0, 1, 0, 1)
        else:
            if max_x == min_x: max_x += 1.0
            if max_y == min_y: max_y += 1.0
            self.world_bounds = (min_x, max_x, min_y, max_y)
            
        # 3. Dibujar
        self._redibujar()

    def _redibujar(self):
        """Calcula coordenadas de pantalla y dibuja items."""
        if not self.G_simplificado or not self.world_pos:
            return
            
        self.delete("all")
        self.nodo_a_items = {}
        self.item_a_nodo = {}
        self.nodo_a_aristas = {n: [] for n in self.G_simplificado.nodes()}
        self.arista_a_nodos = {}
        self.posiciones_screen = {}
        
        width = self.winfo_width() 
        height = self.winfo_height()
        
        if width <= 1: width = 800
        if height <= 1: height = 600
            
        padding = 50
        
        min_x, max_x, min_y, max_y = self.world_bounds
        range_x = max_x - min_x
        range_y = max_y - min_y
        
        avail_w = width - 2 * padding
        avail_h = height - 2 * padding
        
        if range_x == 0: range_x = 1
        if range_y == 0: range_y = 1
        
        scale_x = avail_w / range_x
        scale_y = avail_h / range_y
        scale = min(scale_x, scale_y)
        
        center_win_x = width / 2
        center_win_y = height / 2
        
        center_world_x = (min_x + max_x) / 2
        center_world_y = (min_y + max_y) / 2
        
        # Colores
        colores_lineas = {
            "1": "#E2231A", "2": "#00A651", "3": "#F7931D",
            "4": "#009FE3", "5": "#FFD100", "6": "#8B5A3C",
            "7": "#EE4B9B", "8": "#00A99D", "9": "#9B59B6", "10": "#34495E"
        }
        color_default = "#808080"
        
        # Dibujar Aristas
        for u, v, data in self.G_simplificado.edges(data=True):
            if u not in self.world_pos or v not in self.world_pos:
                continue
            
            ux, uy = self.world_pos[u]
            vx, vy = self.world_pos[v]
            
            sx1 = center_win_x + (ux - center_world_x) * scale
            sy1 = center_win_y - (uy - center_world_y) * scale
            
            sx2 = center_win_x + (vx - center_world_x) * scale
            sy2 = center_win_y - (vy - center_world_y) * scale
            
            color_arista = data.get('color')
            if not color_arista:
                 route_counts = data.get('route_I_counts', {})
                 if route_counts:
                    try:
                        linea = max(route_counts.items(), key=lambda x: x[1])[0]
                        color_arista = colores_lineas.get(str(linea), color_default)
                    except:
                        color_arista = color_default
                 else:
                    color_arista = color_default
            
            width_line = 3 if 'route_I_counts' in data else 2
            
            line_id = self.create_line(sx1, sy1, sx2, sy2, fill=color_arista, width=width_line, tags="arista")
            self.nodo_a_aristas[u].append(line_id)
            self.nodo_a_aristas[v].append(line_id)
            self.arista_a_nodos[line_id] = (u, v)
        
        # Dibujar Nodos
        radio = self.radio_nodo
        for n in self.G_simplificado.nodes():
            if n not in self.world_pos:
                continue
                
            wx, wy = self.world_pos[n]
            sx = center_win_x + (wx - center_world_x) * scale
            sy = center_win_y - (wy - center_world_y) * scale
            
            self.posiciones_screen[n] = (sx, sy)
            self.posiciones[n] = (sx, sy) # Compatibilidad
            
            grado = self.G_simplificado.degree(n)
            color = "white" if grado == 1 else "#FFC107"
            
            tag_oval = f"nodo_{n}"
            tag_text = f"texto_{n}"
            
            oval_id = self.create_oval(sx-radio, sy-radio, sx+radio, sy+radio, 
                                    fill=color, outline="black", width=2, tags=("nodo", tag_oval))
            text_id = self.create_text(sx, sy, text=str(n), fill="black", font=("Arial", 8, "bold"), tags=("texto", tag_text))
            
            self.item_a_nodo[oval_id] = n
            self.item_a_nodo[text_id] = n
            self.nodo_a_items[n] = {'oval': oval_id, 'text': text_id}

    def _get_nodo_center(self, nodo_id):
        """Devuelve (x, y) de pantalla para un nodo."""
        if nodo_id in self.nodo_a_items:
            oval_id = self.nodo_a_items[nodo_id]['oval']
            coords = self.coords(oval_id)
            if coords:
                return (coords[0] + coords[2])/2, (coords[1]+coords[3])/2
        return 0,0

    # Event Handlers (Click, Drag, Zoom, Pan)
    
    def _on_click_nodo(self, event):
        canvas_x = self.canvasx(event.x)
        canvas_y = self.canvasy(event.y)
        try:
            item = self.find_closest(canvas_x, canvas_y)[0]
            nodo_id = self.item_a_nodo.get(item)
            if nodo_id:
                self.nodo_seleccionado = nodo_id
                self.drag_data["x"] = canvas_x
                self.drag_data["y"] = canvas_y
                self.itemconfig(self.nodo_a_items[nodo_id]['oval'], outline="red")
        except IndexError:
            pass

    def _on_drag_nodo(self, event):
        if self.nodo_seleccionado:
            canvas_x = self.canvasx(event.x)
            canvas_y = self.canvasy(event.y)
            dx = canvas_x - self.drag_data["x"]
            dy = canvas_y - self.drag_data["y"]
            
            oval_id = self.nodo_a_items[self.nodo_seleccionado]['oval']
            text_id = self.nodo_a_items[self.nodo_seleccionado]['text']
            self.move(oval_id, dx, dy)
            self.move(text_id, dx, dy)
            
            cx, cy = self._get_nodo_center(self.nodo_seleccionado)
            for line_id in self.nodo_a_aristas[self.nodo_seleccionado]:
                u, v = self.arista_a_nodos[line_id]
                otro = v if u == self.nodo_seleccionado else u
                ox, oy = self._get_nodo_center(otro)
                self.coords(line_id, cx, cy, ox, oy)
            
            self.drag_data["x"] = canvas_x
            self.drag_data["y"] = canvas_y

    def _on_release_nodo(self, event):
        if self.nodo_seleccionado:
            self.itemconfig(self.nodo_a_items[self.nodo_seleccionado]['oval'], outline="black")
            self.nodo_seleccionado = None

    def _on_pan_start(self, event):
        self.scan_mark(event.x, event.y)

    def _on_pan_move(self, event):
        self.scan_dragto(event.x, event.y, gain=1)

    def _on_zoom(self, event):
        scale = 1.0
        if event.delta:
            scale = 1.1 if event.delta > 0 else 0.9
        elif event.num == 4: scale = 1.1
        elif event.num == 5: scale = 0.9
        
        x = self.canvasx(event.x)
        y = self.canvasy(event.y)
        self.scale("all", x, y, scale, scale)

    def _on_drag_start(self, event):
        self.scan_mark(event.x, event.y)

    def _on_drag_move(self, event):
        self.scan_dragto(event.x, event.y, gain=1)

    def _on_double_click_nodo(self, event):
        canvas_x = self.canvasx(event.x)
        canvas_y = self.canvasy(event.y)
        try:
            item = self.find_closest(canvas_x, canvas_y)[0]
            nodo_id = self.item_a_nodo.get(item)
            if nodo_id and self.callback_doble_clic:
                coords = self.pos_geo_cache.get(nodo_id)
                if coords:
                    self.callback_doble_clic(coords[1], coords[0])
        except IndexError:
            pass

    def mostrar_sonar(self, nodo_id_input):
        nodo_id = nodo_id_input
        if nodo_id not in self.nodo_a_items:
            if nodo_id in self.id_mapping:
                nodo_id = self.id_mapping[nodo_id]
            else:
                try:
                    if str(nodo_id) in self.id_mapping:
                         nodo_id = self.id_mapping[str(nodo_id)]
                    elif int(nodo_id) in self.id_mapping:
                         nodo_id = self.id_mapping[int(nodo_id)]
                except: pass
        
        if nodo_id in self.nodo_a_items:
            x, y = self._get_nodo_center(nodo_id)
            r = self.radio_nodo
            sonar_id = self.create_oval(x-r, y-r, x+r, y+r, outline="#0078D7", width=3, stipple="gray50")
            self._animar_sonar(sonar_id, x, y, r, 0)

    def _animar_sonar(self, item_id, x, y, radio_actual, paso):
        if paso >= 20:
            self.delete(item_id)
            return
        progreso = paso / 20
        max_radio = 100
        nuevo_radio = self.radio_nodo + (max_radio - self.radio_nodo) * progreso
        self.coords(item_id, x - nuevo_radio, y - nuevo_radio, x + nuevo_radio, y + nuevo_radio)
        width = max(1, int(4 * (1 - progreso)))
        self.itemconfigure(item_id, width=width)
        self.after(50, lambda: self._animar_sonar(item_id, x, y, nuevo_radio, paso + 1))
