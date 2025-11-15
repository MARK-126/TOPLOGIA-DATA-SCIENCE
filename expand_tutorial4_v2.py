#!/usr/bin/env python3
"""
Script para expandir Tutorial 4 con 2 ejercicios adicionales avanzados
"""

import nbformat as nbf
import sys

def expand_tutorial4():
    """Expande Tutorial 4 con ejercicios 4, 5"""

    # Leer notebook existente
    notebook_path = 'notebooks/04_Mapper_Algorithm_v2.ipynb'
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbf.read(f, as_version=4)

    # Nuevos ejercicios a agregar
    new_exercises = [
        {
            'num': 4,
            'name': 'optimize_mapper_parameters',
            'title': 'Optimizar Parámetros del Mapper',
            'description': '''Los parámetros del **Mapper** (número de intervalos, overlap, método de clustering) afectan significativamente la estructura del grafo resultante. La optimización automática permite encontrar parámetros que mejor capturen la estructura de los datos.

**Aplicación:** Análisis exploratorio de datos cerebrales de alta dimensión, descubrimiento de subtipos de pacientes.

**Tu tarea:** Implementa una función que optimice automáticamente los parámetros del Mapper usando métricas de calidad.''',
            'instructions': '''    """
    Optimiza parámetros del Mapper para maximizar calidad del grafo.

    Parámetros:
    -----------
    data : array, shape (n_samples, n_features)
        Datos de entrada
    filter_function : array, shape (n_samples,)
        Función de filtro pre-calculada
    quality_metric : str
        Métrica a optimizar: 'modularity', 'coverage', 'silhouette'

    Retorna:
    --------
    best_params : dict
        Parámetros óptimos: {'n_intervals', 'overlap', 'n_clusters'}
    best_score : float
        Mejor puntaje de calidad
    mapper_graph : networkx.Graph
        Grafo de Mapper con parámetros óptimos
    """
    # YOUR CODE STARTS HERE
    # (approx. 20-25 lines)
    # Hint 1: Define grid de parámetros (n_intervals: 5-15, overlap: 0.2-0.6, n_clusters: 2-5)
    # Hint 2: Para cada combinación, construye Mapper graph
    # Hint 3: Calcula métrica de calidad (modularity con community detection)
    # Hint 4: Selecciona parámetros con mejor score
    # Hint 5: Retorna parámetros óptimos y grafo final

    # YOUR CODE ENDS HERE''',
            'test_code': '''# Test automático
from notebooks.tda_tests import test_optimize_mapper_parameters
test_optimize_mapper_parameters(optimize_mapper_parameters)'''
        },
        {
            'num': 5,
            'name': 'detect_loops_in_mapper',
            'title': 'Detectar Ciclos Topológicos en Mapper',
            'description': '''Los **ciclos** (loops) en el grafo de Mapper representan características topológicas de dimensión 1 en los datos. Detectar y caracterizar estos ciclos permite identificar estructuras periódicas o cíclicas en datos cerebrales.

**Aplicación:** Identificar oscilaciones neuronales, ciclos de estados cerebrales, patrones recurrentes.

**Tu tarea:** Implementa una función que detecte y caracterice ciclos significativos en el grafo de Mapper.''',
            'instructions': '''    """
    Detecta y caracteriza ciclos (loops) en el grafo de Mapper.

    Parámetros:
    -----------
    mapper_graph : networkx.Graph
        Grafo de Mapper con atributos de nodos
    min_cycle_length : int
        Longitud mínima de ciclo a detectar (default: 3)

    Retorna:
    --------
    cycles : list of lists
        Lista de ciclos detectados (cada ciclo = lista de nodos)
    cycle_features : list of dicts
        Características de cada ciclo:
        - 'length': Longitud del ciclo
        - 'persistence': Persistencia asociada al ciclo
        - 'density': Densidad de datos en el ciclo
    """
    # YOUR CODE STARTS HERE
    # (approx. 15-20 lines)
    # Hint 1: Usa networkx.cycle_basis() para encontrar base de ciclos
    # Hint 2: Filtra ciclos por longitud mínima
    # Hint 3: Para cada ciclo, calcula características:
    #         - length = número de nodos en el ciclo
    #         - density = promedio de densidad de nodos en el ciclo
    #         - persistence = diferencia max-min de filter function en ciclo
    # Hint 4: Ordena ciclos por persistencia (más significativos primero)

    # YOUR CODE ENDS HERE''',
            'test_code': '''# Test automático
from notebooks.tda_tests import test_detect_loops_in_mapper
test_detect_loops_in_mapper(detect_loops_in_mapper)'''
        }
    ]

    # Encontrar índice donde insertar
    insert_idx = None
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == 'markdown' and '## 🎯 Resumen' in ''.join(cell.source):
            insert_idx = i
            break

    if insert_idx is None:
        insert_idx = len(nb.cells)

    # Crear celdas
    new_cells = []

    for ex in new_exercises:
        # Markdown
        md_cell = nbf.v4.new_markdown_cell(f'''---

### Ejercicio {ex['num']} - {ex['name']}

{ex['description']}

**Dificultad:** ⭐⭐⭐ Avanzado
**Tiempo estimado:** 15-20 minutos''')
        new_cells.append(md_cell)

        # Code
        code_cell = nbf.v4.new_code_cell(f'''def {ex['name']}():
{ex['instructions']}''')
        new_cells.append(code_cell)

        # Test
        test_cell = nbf.v4.new_code_cell(ex['test_code'])
        new_cells.append(test_cell)

    # Insertar
    for i, cell in enumerate(new_cells):
        nb.cells.insert(insert_idx + i, cell)

    # Guardar
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)

    print(f"✅ Tutorial 4 expandido: 5 ejercicios totales (agregados 2 nuevos)")
    print(f"   • Ejercicio 4: optimize_mapper_parameters")
    print(f"   • Ejercicio 5: detect_loops_in_mapper")

if __name__ == '__main__':
    try:
        expand_tutorial4()
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
