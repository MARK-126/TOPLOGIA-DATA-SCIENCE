#!/usr/bin/env python3
"""
Script para crear Tutorial 4 v2: Mapper Algorithm (Versión Interactiva)
"""

import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = []

# ========== HEADER ==========
cells.append(nbf.v4.new_markdown_cell("""# Tutorial 4: Algoritmo Mapper (Versión Interactiva)

## Visualización Topológica de Datos Neuronales

**Autor:** MARK-126
**Nivel:** Avanzado
**Tiempo estimado:** 120-150 minutos

---

## 🎯 Objetivos de Aprendizaje

1. ✅ Comprender el algoritmo Mapper
2. ✅ Implementar Mapper desde cero
3. ✅ Aplicar funciones de filtro
4. ✅ Visualizar trayectorias cerebrales
5. ✅ Interpretar grafos neurobiológicamente

---

## ⚠️ Nota sobre Ejercicios

Este notebook contiene **3 ejercicios interactivos** sobre Mapper.

---"""))

# ========== TABLE OF CONTENTS ==========
cells.append(nbf.v4.new_markdown_cell("""<a name='toc'></a>
## 📚 Tabla de Contenidos

- [1 - Setup e Importaciones](#1)
- [2 - Conceptos de Mapper](#2)
    - [Ejercicio 1 - compute_filter_function](#ex-1)
- [3 - Construcción del Grafo](#3)
    - [Ejercicio 2 - build_mapper_graph](#ex-2)
- [4 - Visualización](#4)
    - [Ejercicio 3 - visualize_mapper](#ex-3)
- [5 - Aplicación Neuronal](#5)
- [6 - Resumen](#6)

---"""))

# ========== SECTION 1: SETUP ==========
cells.append(nbf.v4.new_markdown_cell("""<a name='1'></a>
## 1 - Setup e Importaciones

[Volver al índice](#toc)"""))

cells.append(nbf.v4.new_code_cell("""import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import networkx as nx
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from sklearn.datasets import make_swiss_roll

from tda_tests import (
    test_compute_filter_function,
    test_build_mapper_graph,
    test_visualize_mapper
)

plt.style.use('seaborn-v0_8-whitegrid')
np.random.seed(42)
print("✅ Setup completado")"""))

# ========== SECTION 2: EXERCISE 1 ==========
cells.append(nbf.v4.new_markdown_cell("""<a name='2'></a>
## 2 - Algoritmo Mapper: Conceptos

[Volver al índice](#toc)

### ¿Qué es Mapper?

**Mapper** visualiza datos de alta dimensión como un grafo que captura estructura topológica.

### Pasos del Algoritmo:

1. **Función de filtro:** Proyecta datos a 1D o 2D
2. **Cover:** Divide rango en intervalos solapados
3. **Clustering:** Agrupa puntos en cada intervalo
4. **Nerve:** Construye grafo (nodos=clusters, aristas=intersecciones)

<a name='ex-1'></a>
### Ejercicio 1 - compute_filter_function

Implementa diferentes funciones de filtro para Mapper.

**Opciones:**
- **PCA:** Primera componente principal
- **Density:** Densidad local (distancia promedio a vecinos)
- **Coordinate:** Proyección en una coordenada específica"""))

cells.append(nbf.v4.new_code_cell("""# EJERCICIO 1: Funciones de Filtro

def compute_filter_function(data, method='pca', **kwargs):
    \"\"\"
    Calcula función de filtro para Mapper.

    Arguments:
    data -- array (n_samples, n_features)
    method -- 'pca', 'density', o 'coordinate'
    **kwargs -- parámetros adicionales

    Returns:
    filter_values -- array 1D con valores del filtro
    \"\"\"

    if method == 'pca':
        # 1. PCA: Primera componente principal
        # Usar sklearn.decomposition.PCA
        # (approx. 3 lines)
        # YOUR CODE STARTS HERE



        # YOUR CODE ENDS HERE

    elif method == 'density':
        # 2. Densidad: Distancia promedio a k vecinos más cercanos
        # k = kwargs.get('n_neighbors', 10)
        # Calcular matriz de distancias, encontrar vecinos
        # (approx. 6 lines)
        # YOUR CODE STARTS HERE






        # YOUR CODE ENDS HERE

    elif method == 'coordinate':
        # 3. Coordenada: Proyección en dimensión específica
        # coord = kwargs.get('coord_idx', 0)
        # (approx. 1 line)
        # YOUR CODE STARTS HERE

        # YOUR CODE ENDS HERE

    else:
        raise ValueError(f"Método desconocido: {method}")

    return filter_values"""))

cells.append(nbf.v4.new_code_cell("""# Test Ejercicio 1
swiss_roll, colors = make_swiss_roll(n_samples=500, noise=0.1, random_state=42)

filter_pca = compute_filter_function(swiss_roll, method='pca')
filter_density = compute_filter_function(swiss_roll, method='density', n_neighbors=10)

print(f"✅ Filtro PCA: min={filter_pca.min():.2f}, max={filter_pca.max():.2f}")
print(f"   Filtro Density: min={filter_density.min():.2f}, max={filter_density.max():.2f}")
test_compute_filter_function(compute_filter_function)"""))

# ========== SECTION 3: EXERCISE 2 ==========
cells.append(nbf.v4.new_markdown_cell("""<a name='3'></a>
## 3 - Construcción del Grafo de Mapper

[Volver al índice](#toc)

<a name='ex-2'></a>
### Ejercicio 2 - build_mapper_graph

Implementa el algoritmo Mapper completo.

**Pasos:**
1. Aplicar filtro
2. Crear cover (intervalos solapados)
3. Clustering en cada intervalo
4. Construir nerve (grafo)"""))

cells.append(nbf.v4.new_code_cell("""# EJERCICIO 2: Construir Grafo de Mapper

def build_mapper_graph(data, filter_values, n_intervals=10, overlap=0.3):
    \"\"\"
    Construye grafo de Mapper.

    Arguments:
    data -- array (n_samples, n_features)
    filter_values -- valores del filtro (n_samples,)
    n_intervals -- número de intervalos en cover
    overlap -- proporción de solapamiento (0-1)

    Returns:
    G -- grafo de NetworkX
    nodes_data -- diccionario {node_id: indices_de_puntos}
    \"\"\"
    import networkx as nx
    from sklearn.cluster import DBSCAN

    # 1. Crear cover (intervalos solapados)
    # Calcular límites de intervalos
    # (approx. 6 lines)
    # YOUR CODE STARTS HERE






    # YOUR CODE ENDS HERE

    # 2. Clustering en cada intervalo
    # (approx. 12 lines)
    # YOUR CODE STARTS HERE












    # YOUR CODE ENDS HERE

    # 3. Construir grafo (nerve)
    # Nodos = clusters, Aristas = si comparten puntos
    # (approx. 9 lines)
    # YOUR CODE STARTS HERE









    # YOUR CODE ENDS HERE

    return G, nodes_data"""))

cells.append(nbf.v4.new_code_cell("""# Test Ejercicio 2
G_mapper, nodes_mapper = build_mapper_graph(swiss_roll, filter_pca,
                                            n_intervals=15, overlap=0.4)
print(f"✅ Grafo construido")
print(f"   Nodos: {G_mapper.number_of_nodes()}")
print(f"   Aristas: {G_mapper.number_of_edges()}")
test_build_mapper_graph(build_mapper_graph)"""))

# ========== SECTION 4: EXERCISE 3 ==========
cells.append(nbf.v4.new_markdown_cell("""<a name='4'></a>
## 4 - Visualización del Grafo

[Volver al índice](#toc)

<a name='ex-3'></a>
### Ejercicio 3 - visualize_mapper

Visualiza el grafo de Mapper con información significativa.

**Colorear nodos:** Por valor medio del filtro
**Tamaño de nodos:** Por número de puntos
**Layout:** Spring layout de NetworkX"""))

cells.append(nbf.v4.new_code_cell("""# EJERCICIO 3: Visualizar Mapper

def visualize_mapper(G, nodes_data, filter_values, title="Grafo de Mapper"):
    \"\"\"
    Visualiza grafo de Mapper.

    Arguments:
    G -- grafo de NetworkX
    nodes_data -- diccionario {node_id: indices}
    filter_values -- valores del filtro
    title -- título del gráfico

    Returns:
    fig, ax -- figura de matplotlib
    \"\"\"
    fig, ax = plt.subplots(figsize=(10, 8))

    # 1. Calcular layout
    # Usar nx.spring_layout
    # (approx. 1 line)
    # YOUR CODE STARTS HERE

    # YOUR CODE ENDS HERE

    # 2. Calcular tamaños de nodos (proporcional a número de puntos)
    # (approx. 2 lines)
    # YOUR CODE STARTS HERE


    # YOUR CODE ENDS HERE

    # 3. Calcular colores (valor medio del filtro en cada nodo)
    # (approx. 4 lines)
    # YOUR CODE STARTS HERE




    # YOUR CODE ENDS HERE

    # 4. Dibujar grafo
    # (approx. 5 lines)
    # YOUR CODE STARTS HERE





    # YOUR CODE ENDS HERE

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')

    return fig, ax"""))

cells.append(nbf.v4.new_code_cell("""# Test Ejercicio 3
fig, ax = visualize_mapper(G_mapper, nodes_mapper, filter_pca,
                          title="Grafo de Mapper: Swiss Roll")
plt.show()
test_visualize_mapper(visualize_mapper)"""))

# ========== SECTION 5: NEURAL APPLICATION ==========
cells.append(nbf.v4.new_markdown_cell("""<a name='5'></a>
## 5 - Aplicación: Trayectorias Cerebrales

[Volver al índice](#toc)

Aplicamos Mapper a trayectorias de estados cerebrales."""))

cells.append(nbf.v4.new_code_cell("""def generate_brain_trajectory(n_timepoints=400, n_neurons=30):
    \"\"\"Genera trayectoria cíclica de estados cerebrales.\"\"\"
    t = np.linspace(0, 4*np.pi, n_timepoints)

    trajectory = np.zeros((n_timepoints, n_neurons))
    trajectory[:, 0] = 3 * np.cos(t)
    trajectory[:, 1] = 3 * np.sin(t)

    for i in range(2, min(10, n_neurons)):
        trajectory[:, i] = 0.5 * np.sin(t * (i-1) / 2) * np.cos(t * i / 3)

    trajectory[:, 10:] = np.random.randn(n_timepoints, max(0, n_neurons-10)) * 0.3

    # Etiquetas de fase
    phase = (t % (2*np.pi)) / (2*np.pi)
    labels = np.zeros(n_timepoints, dtype=int)
    labels[phase < 0.25] = 0  # Descanso
    labels[(phase >= 0.25) & (phase < 0.5)] = 1  # Atención
    labels[(phase >= 0.5) & (phase < 0.75)] = 2  # Memoria
    labels[phase >= 0.75] = 0

    return trajectory, labels

# Generar y analizar
brain_traj, phase_labels = generate_brain_trajectory(n_timepoints=400, n_neurons=30)
filter_brain = compute_filter_function(brain_traj, method='pca')
G_brain, nodes_brain = build_mapper_graph(brain_traj, filter_brain,
                                          n_intervals=15, overlap=0.4)

print(f"🧠 Trayectoria cerebral: {brain_traj.shape}")
print(f"   Grafo: {G_brain.number_of_nodes()} nodos, {G_brain.number_of_edges()} aristas")

# Visualizar
fig, ax = visualize_mapper(G_brain, nodes_brain, filter_brain,
                          title="Grafo de Mapper: Ciclo Cognitivo")
plt.show()

print("\\n💡 El grafo revela la estructura CÍCLICA de estados cerebrales!")"""))

# ========== SECTION 6: SUMMARY ==========
cells.append(nbf.v4.new_markdown_cell("""<a name='6'></a>
## 6 - Resumen

[Volver al índice](#toc)

<div style="background-color:#e3f2fd; padding:15px; border-left:5px solid #2196f3;">

**💡 Lo que aprendimos:**

- **Mapper** visualiza datos de alta dimensión como grafos
- **Filtros** proyectan datos a dimensiones bajas (PCA, densidad)
- **Cover + Clustering** dividen y agrupan el espacio
- **Nerve** construye grafo de intersecciones
- **Aplicaciones neuronales** revelan ciclos y bifurcaciones

</div>

---

### 🧠 Aplicaciones en Neurociencias

**Trayectorias:** Visualizar evolución de estados cerebrales
**Ciclos cognitivos:** Detectar procesos recurrentes
**Bifurcaciones:** Identificar puntos de decisión
**Espacios de alta dimensión:** Reducir sin perder estructura

---

## 🎉 ¡Excelente trabajo!

Has dominado el algoritmo Mapper para análisis neural.

**Próximo:** Tutorial 5 - Series Temporales EEG

---

**Autor:** MARK-126
**Licencia:** MIT"""))

# ========== CREATE NOTEBOOK ==========
nb['cells'] = cells

with open('notebooks/04_Mapper_Algorithm_v2.ipynb', 'w') as f:
    nbf.write(nb, f)

print("✅ Tutorial 4 v2 creado: notebooks/04_Mapper_Algorithm_v2.ipynb")
