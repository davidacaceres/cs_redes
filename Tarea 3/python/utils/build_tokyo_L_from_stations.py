import json
from pathlib import Path

# Ruta al stations.json del repo tokyo-metro-data
#   https://github.com/Jugendhackt/tokyo-metro-data
STATIONS_JSON = Path("stations.json")
OUTPUT_JSON   = Path("tokyo-L.json")


def main():
    # 1) Cargar datos originales
    with STATIONS_JSON.open(encoding="utf-8") as f:
        tm = json.load(f)

    stations = tm["stations"]  # dict: station_id -> info

    # 2) Crear nodos
    nodes = []
    station_to_node_id = {}

    for idx, (station_id, info) in enumerate(stations.items()):
        station_to_node_id[station_id] = idx

        # Si luego tienes lat/lon, aquí puedes rellenarlas
        node = {
            "id": idx,
            "name": info.get("name_en", station_id),
            "lat": 0.0,              # TODO: sustituir por lat real
            "lon": 0.0,              # TODO: sustituir por lon real
            "original_ids": [station_id],
        }
        nodes.append(node)

    # 3) Crear links (arcos)
    links = []
    seen_edges = set()  # para evitar duplicados (source, target)

    for station_id, info in stations.items():
        src = station_to_node_id[station_id]

        for conn in info.get("connections", []):
            if conn.get("type") != "ride":
                continue  # saltar "walk" u otros

            target_station = conn["target_id"]
            if target_station not in station_to_node_id:
                continue

            dst = station_to_node_id[target_station]

            # Evitar duplicados: si ya vimos (src, dst), no repetir
            if (src, dst) in seen_edges:
                continue
            seen_edges.add((src, dst))

            distance_km = conn.get("distance", 0.0) or 0.0
            duration    = conn.get("duration", -1)

            # Si no hay duración, la aproximamos (30 km/h)
            if duration is None or duration <= 0:
                # tiempo (h) = dist / 30; *3600 -> seg
                duration = int(distance_km / 30.0 * 3600.0)

            # Distancia en metros para parecerse a "d" de Amsterdam
            d_m = int(round(distance_km * 1000.0))

            # Estructura mínima compatible con tu JSON de Ámsterdam
            edge = {
                "shape_id": {},
                "direction_id": {},
                "headsign": {},
                "route_I_counts": {},
                "n_vehicles": 0,
                "duration_avg": float(duration),  # segundos
                "d": d_m,                         # metros
                "source": src,
                "target": dst,
            }
            links.append(edge)

            # Si quieres que el grafo sea claramente bidireccional,
            # puedes añadir también el arco inverso:
            if (dst, src) not in seen_edges:
                seen_edges.add((dst, src))
                edge_rev = edge.copy()
                edge_rev["source"] = dst
                edge_rev["target"] = src
                links.append(edge_rev)

    # 4) Armar grafo final
    graph = {
        "directed": True,
        "multigraph": False,
        "graph": {},
        "nodes": nodes,
        "links": links,
    }

    # 5) Guardar
    with OUTPUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, indent=2)

    print(f"Generado {OUTPUT_JSON} con {len(nodes)} nodos y {len(links)} links.")


if __name__ == "__main__":
    main()
