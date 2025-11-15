#!/usr/bin/env python3
"""
Script para crear Tutorial 3 v2: Conectividad Cerebral (Versión Interactiva)
"""

import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = []

# ========== HEADER ==========
cells.append(nbf.v4.new_markdown_cell("""# Tutorial 3: Conectividad Cerebral con TDA (Versión Interactiva)

## Análisis Topológico de Redes Cerebrales

**Autor:** MARK-126
**Nivel:** Avanzado
**Tiempo estimado:** 150-180 minutos

---

## 🎯 Objetivos de Aprendizaje

1. ✅ Analizar conectomas cerebrales con TDA
2. ✅ Construir matrices de conectividad funcional
3. ✅ Detectar comunidades usando topología
4. ✅ Comparar estados cerebrales topológicamente
5. ✅ Extraer biomarcadores para diagnóstico

---

## ⚠️ Nota sobre Ejercicios

Este notebook contiene **3 ejercicios interactivos** que debes completar.

---"""))

# ========== TABLE OF CONTENTS ==========
cells.append(nbf.v4.new_markdown_cell("""<a name='toc'></a>
## 📚 Tabla de Contenidos

- [1 - Setup e Importaciones](#1)
- [2 - Generación de Datos fMRI](#2)
- [3 - Matriz de Conectividad](#3)
    - [Ejercicio 1 - build_connectivity_matrix](#ex-1)
- [4 - Detección de Comunidades](#4)
    - [Ejercicio 2 - detect_communities_topological](#ex-2)
- [5 - Comparación de Estados](#5)
    - [Ejercicio 3 - compare_states_topologically](#ex-3)
- [6 - Biomarcadores y Resumen](#6)

---"""))

# ========== SECTION 1: SETUP ==========
cells.append(nbf.v4.new_markdown_cell("""<a name='1'></a>
## 1 - Setup e Importaciones

[Volver al índice](#toc)"""))

cells.append(nbf.v4.new_code_cell("""import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from ripser import ripser
from persim import plot_diagrams, bottleneck
import networkx as nx
from scipy.stats import pearsonr
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score
import pandas as pd

from tda_tests import (
    test_build_connectivity_matrix,
    test_detect_communities_topological,
    test_compare_states_topologically
)

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("notebook", font_scale=1.1)
np.random.seed(42)
print("✅ Setup completado")"""))

# ========== SECTION 2: DATA GENERATION ==========
cells.append(nbf.v4.new_markdown_cell("""<a name='2'></a>
## 2 - Generación de Datos fMRI Sintéticos

[Volver al índice](#toc)

### ¿Qué es un Conectoma?

Un **conectoma** es un mapa completo de las conexiones en el cerebro:

- **Conectividad Funcional:** Correlación de actividad entre regiones (fMRI, EEG)
- **Conectividad Estructural:** Conexiones físicas (DTI)
- **Conectividad Efectiva:** Influencia causal entre regiones

### ¿Por qué TDA?

- **Invarianza:** Robusto a ruido y umbrales
- **Multi-escala:** Estructura en todas las escalas
- **Biomarcadores:** Características para diagnóstico"""))

cells.append(nbf.v4.new_code_cell("""def generate_fmri_timeseries(n_regions=50, n_timepoints=200,
                             n_communities=3, noise_level=0.3):
    \"\"\"
    Genera series temporales sintéticas de fMRI con estructura de comunidades.
    \"\"\"
    regions_per_community = n_regions // n_communities
    timeseries = np.zeros((n_regions, n_timepoints))
    labels = np.zeros(n_regions, dtype=int)

    for comm in range(n_communities):
        start_idx = comm * regions_per_community
        end_idx = start_idx + regions_per_community if comm < n_communities - 1 else n_regions

        # Señal común de la comunidad (BOLD response simulada)
        t = np.linspace(0, 4*np.pi, n_timepoints)
        common_signal = np.sin(t + comm * np.pi/3) + 0.5 * np.sin(2*t + comm)

        for i in range(start_idx, end_idx):
            timeseries[i] = common_signal + np.random.randn(n_timepoints) * noise_level
            labels[i] = comm

    # Correlaciones inter-comunidades
    global_signal = np.sin(t) * 0.2
    timeseries += global_signal

    return timeseries, labels

# Generar datos
print("🧠 Generando datos fMRI...\\n")
n_rois = 60
fmri_data, true_labels = generate_fmri_timeseries(
    n_regions=n_rois,
    n_timepoints=250,
    n_communities=3,
    noise_level=0.4
)
print(f"✅ Datos: {fmri_data.shape} (regiones × tiempo)")"""))

# ========== SECTION 3: EXERCISE 1 ==========
cells.append(nbf.v4.new_markdown_cell("""<a name='3'></a>
## 3 - Construcción de Matriz de Conectividad

[Volver al índice](#toc)

<a name='ex-1'></a>
### Ejercicio 1 - build_connectivity_matrix

Construye matriz de conectividad funcional y filtra conexiones débiles.

**Pasos:**
1. Calcular correlación de Pearson entre todas las regiones
2. Aplicar umbral para eliminar correlaciones débiles
3. Construir matriz de distancias topológicas
4. Calcular homología persistente

**Fórmulas:**
- Correlación: $C_{ij} = \\text{corr}(X_i(t), X_j(t))$
- Distancia: $d_{ij} = 1 - |C_{ij}|$"""))

cells.append(nbf.v4.new_code_cell("""# EJERCICIO 1: Construir Matriz de Conectividad

def build_connectivity_matrix(timeseries, threshold=0.3):
    \"\"\"
    Construye matriz de conectividad funcional con filtrado y análisis topológico.

    Arguments:
    timeseries -- array (n_regions, n_timepoints)
    threshold -- umbral para filtrar correlaciones débiles

    Returns:
    conn_matrix -- matriz de conectividad filtrada
    diagrams -- diagramas de persistencia
    \"\"\"
    n_regions = timeseries.shape[0]

    # 1. Calcular correlaciones entre todas las regiones
    # Usar scipy.stats.pearsonr en un loop doble
    # (approx. 6 lines)
    # YOUR CODE STARTS HERE






    # YOUR CODE ENDS HERE

    # 2. Aplicar umbral: valores < threshold → 0
    # (approx. 2 lines)
    # YOUR CODE STARTS HERE


    # YOUR CODE ENDS HERE

    # 3. Convertir a distancias: d_ij = 1 - |C_ij|
    # (approx. 3 lines)
    # YOUR CODE STARTS HERE



    # YOUR CODE ENDS HERE

    # 4. Calcular persistencia con matriz de distancia
    # Usar ripser con distance_matrix=True
    # (approx. 2 lines)
    # YOUR CODE STARTS HERE


    # YOUR CODE ENDS HERE

    return conn_matrix, diagrams"""))

cells.append(nbf.v4.new_code_cell("""# Test Ejercicio 1
conn_matrix, diagrams = build_connectivity_matrix(fmri_data, threshold=0.3)
print(f"✅ Matriz: {conn_matrix.shape}")
print(f"   H₀: {len(diagrams[0])} componentes")
print(f"   H₁: {len(diagrams[1])} ciclos")
test_build_connectivity_matrix(build_connectivity_matrix)"""))

# ========== SECTION 4: EXERCISE 2 ==========
cells.append(nbf.v4.new_markdown_cell("""<a name='4'></a>
## 4 - Detección de Comunidades Funcionales

[Volver al índice](#toc)

<a name='ex-2'></a>
### Ejercicio 2 - detect_communities_topological

Detecta comunidades funcionales usando características topológicas.

**Estrategia:**
1. Extraer características topológicas de cada región (persistencia local)
2. Usar clustering espectral en matriz de conectividad
3. Validar con métricas de calidad (ARI)

**Métricas:**
- **ARI (Adjusted Rand Index):** Mide similitud con ground truth
- **NMI (Normalized Mutual Information):** Información compartida"""))

cells.append(nbf.v4.new_code_cell("""# EJERCICIO 2: Detectar Comunidades

def detect_communities_topological(conn_matrix, n_clusters=3, true_labels=None):
    \"\"\"
    Detecta comunidades usando clustering espectral y evalúa calidad.

    Arguments:
    conn_matrix -- matriz de conectividad (n_regions, n_regions)
    n_clusters -- número de comunidades
    true_labels -- etiquetas verdaderas (opcional, para evaluación)

    Returns:
    detected_labels -- etiquetas asignadas a cada región
    ari_score -- Adjusted Rand Index (si true_labels dado)
    \"\"\"

    # 1. Preparar matriz de afinidad (valores absolutos)
    # (approx. 1 line)
    # YOUR CODE STARTS HERE

    # YOUR CODE ENDS HERE

    # 2. Aplicar Spectral Clustering
    # Usar sklearn.cluster.SpectralClustering con affinity='precomputed'
    # (approx. 3 lines)
    # YOUR CODE STARTS HERE



    # YOUR CODE ENDS HERE

    # 3. Evaluar con ARI si hay etiquetas verdaderas
    # (approx. 4 lines)
    # YOUR CODE STARTS HERE




    # YOUR CODE ENDS HERE

    return detected_labels, ari_score"""))

cells.append(nbf.v4.new_code_cell("""# Test Ejercicio 2
detected_labels, ari = detect_communities_topological(conn_matrix, n_clusters=3,
                                                      true_labels=true_labels)
print(f"✅ Comunidades detectadas")
print(f"   ARI: {ari:.3f}")
print(f"   (1.0 = detección perfecta)")
test_detect_communities_topological(detect_communities_topological)"""))

# ========== SECTION 5: EXERCISE 3 ==========
cells.append(nbf.v4.new_markdown_cell("""<a name='5'></a>
## 5 - Comparación de Estados Cerebrales

[Volver al índice](#toc)

<a name='ex-3'></a>
### Ejercicio 3 - compare_states_topologically

Compara diferentes estados cerebrales usando distancias topológicas.

**Estados a comparar:**
- Reposo (resting state)
- Tarea cognitiva
- Sueño

**Distancia:** Bottleneck distance entre diagramas de persistencia H₁"""))

cells.append(nbf.v4.new_code_cell("""# EJERCICIO 3: Comparar Estados

def compare_states_topologically(states_dict):
    \"\"\"
    Compara estados cerebrales usando distancia de bottleneck.

    Arguments:
    states_dict -- diccionario {state_name: timeseries}

    Returns:
    distance_matrix -- matriz de distancias entre estados
    all_diagrams -- diagramas de cada estado
    \"\"\"
    from persim import bottleneck

    state_names = list(states_dict.keys())
    n_states = len(state_names)
    all_diagrams = {}

    # 1. Calcular diagrama de persistencia para cada estado
    # Usar build_connectivity_matrix con cada timeseries
    # (approx. 4 lines)
    # YOUR CODE STARTS HERE




    # YOUR CODE ENDS HERE

    # 2. Calcular matriz de distancias de bottleneck entre H₁
    # (approx. 7 lines)
    # YOUR CODE STARTS HERE







    # YOUR CODE ENDS HERE

    return distance_matrix, all_diagrams"""))

cells.append(nbf.v4.new_code_cell("""# Generar estados diferentes
states = {
    'Reposo': generate_fmri_timeseries(n_rois, 250, n_communities=2, noise_level=0.5)[0],
    'Tarea': generate_fmri_timeseries(n_rois, 250, n_communities=4, noise_level=0.3)[0],
    'Sueño': generate_fmri_timeseries(n_rois, 250, n_communities=1, noise_level=0.2)[0]
}

# Test Ejercicio 3
dist_matrix, all_diagrams = compare_states_topologically(states)
print(f"✅ Matriz de distancias:\\n{dist_matrix}")
test_compare_states_topologically(compare_states_topologically)"""))

# ========== SECTION 6: SUMMARY ==========
cells.append(nbf.v4.new_markdown_cell("""<a name='6'></a>
## 6 - Biomarcadores y Resumen

[Volver al índice](#toc)

### 🩺 Biomarcadores Topológicos

Características que pueden usarse clínicamente:

**H₀ (Componentes):**
- Integración funcional
- Fragmentación de red

**H₁ (Ciclos):**
- Circuitos de retroalimentación
- Complejidad funcional
- **Alzheimer:** Reducción significativa

**H₂ (Cavidades):**
- Organización jerárquica
- **Esquizofrenia:** Alteraciones estructurales

---

<div style="background-color:#e3f2fd; padding:15px; border-left:5px solid #2196f3;">

**💡 Lo que aprendimos:**

- **Conectomas** se representan como matrices de conectividad
- **TDA** captura estructura multi-escala de redes cerebrales
- **Comunidades funcionales** se detectan con topología
- **Estados cerebrales** se comparan con distancias topológicas
- **Biomarcadores** tienen aplicación clínica directa

</div>

---

## 🎉 ¡Felicitaciones!

Dominas el análisis topológico de conectividad cerebral.

**Próximo:** Tutorial 4 - Algoritmo Mapper

---

**Autor:** MARK-126
**Licencia:** MIT"""))

# ========== CREATE NOTEBOOK ==========
nb['cells'] = cells

with open('notebooks/03_Conectividad_Cerebral_v2.ipynb', 'w') as f:
    nbf.write(nb, f)

print("✅ Tutorial 3 v2 creado: notebooks/03_Conectividad_Cerebral_v2.ipynb")
