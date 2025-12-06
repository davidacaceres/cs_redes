
import os
import hashlib
from pathlib import Path

def get_file_hash(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        h.update(f.read())
    return h.hexdigest()

def limpiar_duplicados():
    base_dir = Path(r"c:\Users\dcaceres\Documents\Magister\Trimestre 2\CS Redes\Tarea 3\python\sitio\banderas")
    if not base_dir.exists():
        print(f"No existe: {base_dir}")
        return

    files = list(base_dir.glob("*.png"))
    print(f"Procesando {len(files)} archivos...")

    # Group by hash
    by_hash = {}
    for f in files:
        h = get_file_hash(f)
        if h not in by_hash:
            by_hash[h] = []
        by_hash[h].append(f)

    deleted_count = 0
    for h, group in by_hash.items():
        if len(group) > 1:
            # Sort to pick best: Prefer containing '-' (country-city), shortest?
            # Prefer 'country-city.png', then 'city.png', avoid '*-L.png', 'kujala_*.png'
            
            def score_name(p):
                name = p.stem.lower()
                s = 0
                if "kujala_" in name: s -= 10
                if "-l" in name: s -= 5
                if "-" in name and not "-l" in name: s += 10 # country-city
                return s

            group.sort(key=score_name, reverse=True)
            keeper = group[0]
            duplicates = group[1:]
            
            print(f"Manteniendo: {keeper.name}")
            for d in duplicates:
                print(f"  Eliminando: {d.name}")
                try:
                    os.remove(d)
                    deleted_count += 1
                except Exception as e:
                    print(f"  Error eliminando {d.name}: {e}")

    print(f"Limpieza completada. {deleted_count} archivos eliminados.")

if __name__ == "__main__":
    limpiar_duplicados()
