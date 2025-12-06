
import os
import requests
from pathlib import Path
import time

# Mapping of network names to Spanish Wikipedia titles (City, Metro)
# Format: "network_name": ("City_Title", "Metro_Title")
WIKI_MAPPING = {
    "amsterdam-L": ("Ámsterdam", "Metro_de_Ámsterdam"),
    "athens-L": ("Atenas", "Metro_de_Atenas"),
    "atlanta-L": ("Atlanta", "MARTA"),
    "baltimore-L": ("Baltimore", "Metro_de_Baltimore"),
    "berlin-L": ("Berlín", "Metro_de_Berlín"),
    "bilbao-L": ("Bilbao", "Metro_de_Bilbao"),
    "boston-L": ("Boston", "Metro_de_Boston"),
    "brussels-L": ("Bruselas", "Metro_de_Bruselas"),
    "budapest-L": ("Budapest", "Metro_de_Budapest"),
    "buenosaires-L": ("Buenos_Aires", "Subte_de_Buenos_Aires"),
    "cairo-L": ("El_Cairo", "Metro_de_El_Cairo"),
    "chicago-L": ("Chicago", "Metro_de_Chicago"),
    "cleveland-L": ("Cleveland", "RTA_Rapid_Transit"),
    "copenhagen-L": ("Copenhague", "Metro_de_Copenhague"),
    "dubai-L": ("Dubái", "Metro_de_Dubái"),
    "genoa-L": ("Génova", "Metro_de_Génova"),
    "helsinki-L": ("Helsinki", "Metro_de_Helsinki"),
    "hyderabad-L": ("Hyderabad_(India)", "Metro_de_Hyderabad"),
    "kobe-L": ("Kōbe", "Metro_Municipal_de_Kobe"),
    "kochi-L": ("Kochi_(India)", "Metro_de_Kochi"),
    "lille-L": ("Lille", "Metro_de_Lille"),
    "lisbon-L": ("Lisboa", "Metro_de_Lisboa"),
    "london-L": ("Londres", "Metro_de_Londres"),
    "losangeles-L": ("Los_Ángeles", "Metro_de_Los_Ángeles"),
    "lyon-L": ("Lyon", "Metro_de_Lyon"),
    "madrid-L": ("Madrid", "Metro_de_Madrid"),
    "malaga-L": ("Málaga", "Metro_de_Málaga"),
    "marseille-L": ("Marsella", "Metro_de_Marsella"),
    "milan-L": ("Milán", "Metro_de_Milán"),
    "montreal-L": ("Montreal", "Metro_de_Montreal"),
    "naples-L": ("Nápoles", "Metro_de_Nápoles"),
    "newyork-L": ("Nueva_York", "Metro_de_Nueva_York"),
    "nuremberg-L": ("Núremberg", "Metro_de_Núremberg"),
    "oslo-L": ("Oslo", "Metro_de_Oslo"),
    "paris-L": ("París", "Metro_de_París"),
    "philadelphia-L": ("Filadelfia", "Metro_de_Filadelfia"),
    "prague-L": ("Praga", "Metro_de_Praga"),
    "rennes-L": ("Rennes", "Metro_de_Rennes"),
    "rome-L": ("Roma", "Metro_de_Roma"),
    "rotterdam-L": ("Róterdam", "Metro_de_Róterdam"),
    "sanfrancisco-L": ("San_Francisco_(California)", "BART"),
    "santiago-L": ("Santiago_de_Chile", "Metro_de_Santiago"),
    "stockholm-L": ("Estocolmo", "Metro_de_Estocolmo"),
    "tokyo-L": ("Tokio", "Metro_de_Tokio"),
    "toronto-L": ("Toronto", "Metro_de_Toronto"),
    "toulouse-L": ("Toulouse", "Metro_de_Toulouse"),
    "turin-L": ("Turín", "Metro_de_Turín"),
    "valencia-L": ("Valencia", "Metrovalencia"),
    "vancouver-L": ("Vancouver", "SkyTrain_(Vancouver)"),
    "vienna-L": ("Viena", "Metro_de_Viena"),
    "warsaw-L": ("Varsovia", "Metro_de_Varsovia"),
    "washington-L": ("Washington_D._C.", "Metro_de_Washington")
}

def get_wiki_summary(title):
    """Fetches the summary of a Wikipedia page (Spanish)."""
    url = f"https://es.wikipedia.org/api/rest_v1/page/summary/{title}"
    try:
        response = requests.get(url, headers={'User-Agent': 'MetroNetworkAnalysis/1.0'}, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data.get('extract', 'No hay descripción disponible.')
        else:
            print(f"[WARN] Could not fetch summary for {title}: {response.status_code}")
            return None
    except Exception as e:
        print(f"[ERROR] Error fetching {title}: {e}")
        return None

def download_info():
    base_dir = Path("sitio/info")
    base_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading city and metro info (Spanish) to {base_dir}...")
    
    for network, (city_title, metro_title) in WIKI_MAPPING.items():
        output_path = base_dir / f"{network}.md"
        
        print(f"Processing {network}...")
        
        city_summary = get_wiki_summary(city_title)
        metro_summary = get_wiki_summary(metro_title)
        
        content = ""
        if city_summary:
            content += f"### Sobre la Ciudad: {city_title.replace('_', ' ')}\n\n"
            content += f"{city_summary}\n\n"
        
        if metro_summary:
            content += f"### Sobre la Red de Metro: {metro_title.replace('_', ' ')}\n\n"
            content += f"{metro_summary}\n"
            
        if content:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"[OK] Saved info for {network}")
        else:
            print(f"[FAIL] No info found for {network}")
            
        # Be nice to Wikipedia API
        time.sleep(0.5)

if __name__ == "__main__":
    download_info()
