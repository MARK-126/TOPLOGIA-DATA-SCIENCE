#!/usr/bin/env python3
"""
Script para expandir Tutorial 3 con 3 ejercicios adicionales avanzados
"""

import nbformat as nbf
import sys

def expand_tutorial3():
    """Expande Tutorial 3 con ejercicios 4, 5, 6"""

    # Leer notebook existente
    notebook_path = 'notebooks/03_Conectividad_Cerebral_v2.ipynb'
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbf.read(f, as_version=4)

    # Nuevos ejercicios a agregar
    new_exercises = [
        {
            'num': 4,
            'name': 'compute_graph_features',
            'title': 'Extraer Características de Grafo de Conectividad',
            'description': '''Las **características de grafo** (clustering coefficient, betweenness centrality, modularity) combinadas con **características topológicas** proporcionan una descripción completa de la organización cerebral.

**Aplicación clínica:** Identificar biomarcadores de red en Alzheimer, autismo, esquizofrenia.

**Tu tarea:** Implementa una función que extraiga características de grafo y TDA de una matriz de conectividad.''',
            'instructions': '''    """
    Extrae características combinadas de grafo y TDA.

    Parámetros:
    -----------
    connectivity_matrix : array, shape (n_nodes, n_nodes)
        Matriz de conectividad funcional
    threshold : float
        Umbral para binarizar matriz (default: 0.3)

    Retorna:
    --------
    features : dict
        Diccionario con características:
        - 'clustering_coeff': Coeficiente de clustering promedio
        - 'path_length': Longitud de camino promedio
        - 'modularity': Modularidad de la red
        - 'betti_0': Número de componentes conectadas
        - 'betti_1': Número de ciclos
        - 'persistence_entropy': Entropía de persistencia
    """
    # YOUR CODE STARTS HERE
    # (approx. 15-20 lines)
    # Hint 1: Binariza matriz con threshold para crear grafo
    # Hint 2: Usa networkx para calcular clustering_coefficient, average_shortest_path_length
    # Hint 3: Usa community.modularity para calcular modularity
    # Hint 4: Calcula homología persistente con ripser
    # Hint 5: Extrae Betti numbers del diagrama de persistencia
    # Hint 6: Calcula persistence entropy

    # YOUR CODE ENDS HERE''',
            'test_code': '''# Test automático
from notebooks.tda_tests import test_compute_graph_features
test_compute_graph_features(compute_graph_features)'''
        },
        {
            'num': 5,
            'name': 'find_critical_nodes',
            'title': 'Identificar Nodos Críticos usando TDA',
            'description': '''Los **nodos críticos** son regiones cerebrales cuya eliminación causa cambios topológicos significativos en la red. Estos nodos son candidatos para targets terapéuticos o regiones vulnerables en enfermedades.

**Aplicación:** Planificación quirúrgica en epilepsia, identificación de hubs patológicos.

**Tu tarea:** Implementa una función que identifique nodos críticos mediante análisis topológico de ablación.''',
            'instructions': '''    """
    Identifica nodos críticos mediante ablación topológica.

    Parámetros:
    -----------
    connectivity_matrix : array, shape (n_nodes, n_nodes)
        Matriz de conectividad
    top_k : int
        Número de nodos más críticos a retornar (default: 5)

    Retorna:
    --------
    critical_nodes : array
        Índices de nodos críticos (ordenados por criticidad)
    criticality_scores : array
        Puntajes de criticidad para cada nodo
    """
    # YOUR CODE STARTS HERE
    # (approx. 12-18 lines)
    # Hint 1: Calcula características topológicas de la red original
    # Hint 2: Para cada nodo i, elimina fila/columna i y recalcula características
    # Hint 3: Criticidad = cambio en características topológicas (Wasserstein o Betti)
    # Hint 4: Retorna top_k nodos con mayor criticidad

    # YOUR CODE ENDS HERE''',
            'test_code': '''# Test automático
from notebooks.tda_tests import test_find_critical_nodes
test_find_critical_nodes(find_critical_nodes)'''
        },
        {
            'num': 6,
            'name': 'track_connectivity_evolution',
            'title': 'Rastrear Evolución de Conectividad Temporal',
            'description': '''El **seguimiento temporal** de cambios en conectividad cerebral permite estudiar plasticidad, aprendizaje, progresión de enfermedades, o efectos de intervenciones.

**Aplicación:** Monitoreo de rehabilitación post-stroke, evaluación de tratamientos farmacológicos.

**Tu tarea:** Implementa una función que rastree cómo evoluciona la topología de conectividad a lo largo del tiempo.''',
            'instructions': '''    """
    Rastrea evolución temporal de conectividad usando TDA.

    Parámetros:
    -----------
    time_series : array, shape (n_timepoints, n_channels, n_samples)
        Series temporales multi-canal en diferentes timepoints
    window_indices : list
        Índices que definen ventanas temporales

    Retorna:
    --------
    evolution_metrics : dict
        - 'betti_evolution': Evolución de números de Betti
        - 'persistence_evolution': Evolución de suma de persistencias
        - 'connectivity_strength': Fuerza de conectividad promedio
        - 'topological_transitions': Índices donde ocurren cambios grandes
    """
    # YOUR CODE STARTS HERE
    # (approx. 18-25 lines)
    # Hint 1: Para cada timepoint, construye matriz de conectividad
    # Hint 2: Calcula homología persistente en cada timepoint
    # Hint 3: Extrae Betti numbers y suma de persistencias
    # Hint 4: Detecta transiciones donde hay cambios > threshold
    # Hint 5: Retorna métricas de evolución temporal

    # YOUR CODE ENDS HERE''',
            'test_code': '''# Test automático
from notebooks.tda_tests import test_track_connectivity_evolution
test_track_connectivity_evolution(track_connectivity_evolution)'''
        }
    ]

    # Encontrar índice donde insertar (después del último ejercicio)
    insert_idx = None
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == 'markdown' and '## 🎯 Resumen' in ''.join(cell.source):
            insert_idx = i
            break

    if insert_idx is None:
        insert_idx = len(nb.cells)

    # Crear celdas para cada ejercicio
    new_cells = []

    for ex in new_exercises:
        # Celda markdown
        md_cell = nbf.v4.new_markdown_cell(f'''---

### Ejercicio {ex['num']} - {ex['name']}

{ex['description']}

**Dificultad:** ⭐⭐⭐ Avanzado
**Tiempo estimado:** 15-20 minutos''')
        new_cells.append(md_cell)

        # Celda de código
        code_cell = nbf.v4.new_code_cell(f'''def {ex['name']}():
{ex['instructions']}''')
        new_cells.append(code_cell)

        # Celda de test
        test_cell = nbf.v4.new_code_cell(ex['test_code'])
        new_cells.append(test_cell)

    # Insertar nuevas celdas
    for i, cell in enumerate(new_cells):
        nb.cells.insert(insert_idx + i, cell)

    # Guardar notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)

    print(f"✅ Tutorial 3 expandido: 6 ejercicios totales (agregados 3 nuevos)")
    print(f"   • Ejercicio 4: compute_graph_features")
    print(f"   • Ejercicio 5: find_critical_nodes")
    print(f"   • Ejercicio 6: track_connectivity_evolution")

if __name__ == '__main__':
    try:
        expand_tutorial3()
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
