
import os
import requests
from pathlib import Path

# Mapping of city names (from filenames) to ISO 3166-1 alpha-2 country codes
CITY_COUNTRY_MAP = {
    "amsterdam": "nl",
    "athens": "gr",
    "atlanta": "us",
    "baltimore": "us",
    "berlin": "de",
    "bilbao": "es",
    "boston": "us",
    "brussels": "be",
    "budapest": "hu",
    "buenosaires": "ar",
    "cairo": "eg",
    "chicago": "us",
    "cleveland": "us",
    "copenhagen": "dk",
    "dubai": "ae",
    "genoa": "it",
    "helsinki": "fi",
    "hyderabad": "in",
    "kobe": "jp",
    "kochi": "in",
    "lille": "fr",
    "lisbon": "pt",
    "london": "gb",
    "losangeles": "us",
    "lyon": "fr",
    "madrid": "es",
    "malaga": "es",
    "marseille": "fr",
    "milan": "it",
    "montreal": "ca",
    "naples": "it",
    "newyork": "us",
    "nuremberg": "de",
    "oslo": "no",
    "paris": "fr",
    "philadelphia": "us",
    "prague": "cz",
    "rennes": "fr",
    "rome": "it",
    "rotterdam": "nl",
    "sanfrancisco": "us",
    "santiago": "cl",
    "stockholm": "se",
    "tokyo": "jp",
    "toronto": "ca",
    "toulouse": "fr",
    "turin": "it",
    "valencia": "es",
    "vancouver": "ca",
    "vienna": "at",
    "warsaw": "pl",
    "washington": "us"
}

def download_flags():
    base_dir = Path("sitio/banderas")
    base_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading flags to {base_dir}...")
    
    for city, country_code in CITY_COUNTRY_MAP.items():
        network_name = f"{city}-L"
        url = f"https://flagcdn.com/w640/{country_code}.png"
        output_path = base_dir / f"{network_name}.png"
        
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                print(f"[OK] Downloaded flag for {network_name} ({country_code})")
            else:
                print(f"[ERROR] Failed to download flag for {network_name}: Status {response.status_code}")
        except Exception as e:
            print(f"[ERROR] Exception downloading flag for {network_name}: {e}")

if __name__ == "__main__":
    download_flags()
