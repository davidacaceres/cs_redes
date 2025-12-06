
import os
import google.generativeai as genai
from typing import Dict, Any

class GeminiAnalyzer:
    def __init__(self, api_key: str = None):
        """
        Inicializa el cliente de Gemini.
        
        Si no se proporciona una clave API, intenta obtenerla de la variable de entorno "GEMINI_API_KEY".
        """
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-pro-latest')
        else:
            self.model = None

    def set_api_key(self, api_key: str):
        self.api_key = api_key
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-pro-latest')

    def analyze_network(self, metrics: Dict[str, Any], city_name: str) -> str:
        if not self.model:
             return "Error: API Key no configurada. Por favor ingrese su clave de Gemini."

        prompt = self._build_prompt(metrics, city_name)
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error al consultar Gemini: {str(e)}"

    def _build_prompt(self, metrics: Dict[str, Any], city_name: str) -> str:
        import json
        
        data = {
            "network_info": {
                "name": city_name,
                "type": "Metro Network"
            },
            "topology": {
                "nodes_N": metrics.get("N", 0),
                "edges_L": metrics.get("L", 0),
                "assortativity_r": metrics.get("r̄ᵀ", 0.0),
                "clustering_coefficient_C_G": metrics.get("CC_G", 0.0),
                "modularity_M_G": metrics.get("M_G", 0.0)
            },
            "efficiency": {
                "global_efficiency_E": metrics.get("E[1/H]", 0.0),
                "average_path_length_L": metrics.get("longitud_camino_promedio", 0.0), 
                "reliability_Rel_G": metrics.get("Rel_G", 0.0)
            },
            "spectral_properties": {
                "algebraic_connectivity_mu_n_1": metrics.get("μ̄ₙ₋₁", 0.0),
                "spectral_radius_lambda_star": metrics.get("λ̄*", 0.0), 
                "kappa": metrics.get("Kappa", 0.0),
                "inverse_kappa": metrics.get("1/κ", 0.0)
            },
            "robustness": {
                "f90_degree": metrics.get("f₉₀%-degree", 0.0),
                "f90_random": metrics.get("f₉₀%-random", 0.0),
                "fc_degree": metrics.get("f_c-degree", 0.0),
                "fc_random": metrics.get("f_c-random", 0.0),
                "robustness_area": metrics.get("Area", 0.0)
            }
        }
        
        json_str = json.dumps(data, indent=2)
        
        prompt = f"""
Actúa como un experto en Teoría de Redes Complejas y Análisis de Redes de Transporte.
Realiza un análisis breve pero profundo de la robustez y características topológicas de la red de metro de **{city_name}** basándote en los siguientes datos en formato JSON:

```json
{json_str}
```

Por favor, interpreta estas métricas enfocándote en:
1. **Topología**: Qué nos dicen N, L y la asortatividad sobre la forma de la red.
2. **Eficiencia**: Analiza la eficiencia global y fiabilidad.
3. **Propiedades Espectrales**: Qué implican la conectividad algebraica y el radio espectral para la dispersión de fallos.
4. **Robustez**: Interpreta los umbrales de percolación (f90, fc) y el área de robustez.


Tus objetivos son:
1. Evaluar la eficiencia global.
2. Comentar sobre la vulnerabilidad basada en los hubs.
3. Dar una conclusión breve de 3 líneas.


Entrega el resultado en formato Markdown limpio. Sé conciso y directo en tus conclusiones.
"""
        return prompt
