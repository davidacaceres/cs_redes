# Analizador de Robustez de Redes de Metro

Esta aplicación permite cargar, analizar y visualizar redes complejas de metros del mundo, enfocándose en el cálculo de métricas de robustez y resiliencia ante fallos.

## Características Principales

### 1. Gestión de Datos
*   **Carga de Dataset Kujala**: Soporte para el dataset "Collection of 33 metro networks" (Kujala et al.).
*   **Carga de Dataset Metro51**: Soporte para formatos JSON estandarizados de 51 redes.
*   **Preprocesamiento**: Limpieza de nombres, normalización de coordenadas y conversión a grafos `networkx`.

### 2. Análisis Topológico y Robusto
Calcula métricas avanzadas de teoría de grafos y robustez:
*   **Métricas Básicas**: $N$ (Nodos), $L$ (Enlaces), Densidad.
*   **Conectividad**: Componente Gigante, Eficiencia Global ($E[1/H]$).
*   **Robustez Estructural**:
    *   $\kappa$ (Kappa): Diversidad de grados.
    *   $r^T$: Robustez topológica (densidad de ciclos).
    *   $\lambda^*$: Conectividad natural.
    *   $\mu_{N-1}$: Conectividad algebraica.
*   **Simulación de Fallos**:
    *   **Fallos Aleatorios**: Eliminación progresiva de nodos al azar.
    *   **Ataques Dirigidos**: Eliminación de nodos según su grado (hub attacks).
    *   Cálculo de umbrales de percolación ($f_{90\%}$, $f_c$).

### 3. Visualización Interactiva
*   **Mapa Geográfico**: Visualización sobre mapas reales (OpenStreetMap/Satélite) con ubicación de estaciones.
*   **Grafo Topológico**: Representación abstracta "spring layout" de la red.
*   **Gráfico Radar**: Comparación visual de múltiples métricas normalizadas (estilo "Atlas of Urban Subway Networks").
*   **Curvas de Robustez**: Gráficos de línea mostrando la degradación del Componente Gigante ante fallos.

### 4. Generación de Reportes
*   **Reportes Individuales**: Exportación a HTML autocontenido que incluye:
    *   Tabla de métricas detallada.
    *   Información enciclopédica de la ciudad.
    *   Gráficos estáticos embebidos (Radar, Robustez, Mapas).
*   **Sitio Web Completo**: Generación batch de un sitio web estático navegable con índice y páginas por red.

## Requisitos del Sistema

*   **Python 3.8+**
*   **Bibliotecas**:
    *   `networkx`: Procesamiento de grafos.
    *   `matplotlib`: Generación de gráficos estáticos.
    *   `pandas`: Manejo de datos tabulares.
    *   `numpy`, `scipy`: Cálculos numéricos y álgebra lineal.
    *   `tkinter`: Interfaz gráfica de usuario (incluido en Python estándar).
    *   `contextily` (Opcional): Para mapas base geográficos.

## Uso

1.  **Ejecutar la interfaz**:
    ```bash
    python interfaz_grafica.py
    ```

2.  **Seleccionar un Dataset**: Use el menú desplegable para elegir entre "Kujala" o "Metro51".
3.  **Seleccionar una Ciudad**: Elija una red específica (ej. "London", "Santiago").
4.  **Generar Análisis**: Haga clic en "Generar Análisis" para calcular métricas y visualizar.
5.  **Explorar Pestañas**:
    *   *Info General*: Métricas numéricas y Gráfico Radar.
    *   *Visualización*: Mapa interactivo.
    *   *Robustez*: Curvas de ataque y fallo.
    *   *Componentes*: Análisis de fragmentación.
6.  **Exportar**: Use el botón "Exportar Reporte" para guardar los resultados en HTML.

## Estructura del Proyecto

*   `interfaz_grafica.py`: Punto de entrada y lógica de la GUI.
*   `preparar_redes.py`: Módulos de carga y transformación de datos.
*   `procesar_redes.py`: Motor de cálculo de métricas y simulaciones.
*   `visualizacion.py`: Lógica de generación de reportes HTML.
*   `data/`: Directorio para los datasets (debe contener `kujala` y `51_metro_networks`).
