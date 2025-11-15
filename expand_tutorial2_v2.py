#!/usr/bin/env python3
"""
Script para expandir Tutorial 2 con 3 ejercicios adicionales avanzados
"""

import nbformat as nbf
import sys

def expand_tutorial2():
    """Expande Tutorial 2 con ejercicios 5, 6, 7"""

    # Leer notebook existente
    notebook_path = 'notebooks/02_Homologia_Persistente_Avanzada_v2.ipynb'
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbf.read(f, as_version=4)

    # Nuevos ejercicios a agregar
    new_exercises = [
        {
            'num': 5,
            'name': 'compute_wasserstein_distance',
            'title': 'Calcular Distancia de Wasserstein entre Diagramas',
            'description': '''La **distancia de Wasserstein** (también conocida como Earth Mover's Distance) cuantifica la diferencia entre dos diagramas de persistencia. Es más informativa que la distancia de bottleneck porque considera todos los puntos, no solo el peor caso.

**Aplicación clínica:** Comparar diagramas de persistencia de diferentes estados cerebrales para clasificación automática.

**Tu tarea:** Implementa una función que calcule la distancia de Wasserstein entre dos diagramas de persistencia.''',
            'instructions': '''    """
    Calcula la distancia de Wasserstein entre dos diagramas de persistencia.

    Parámetros:
    -----------
    dgm1, dgm2 : array-like, shape (n_points, 2)
        Diagramas de persistencia (birth, death)

    Retorna:
    --------
    distance : float
        Distancia de Wasserstein entre los diagramas
    """
    # YOUR CODE STARTS HERE
    # (approx. 8-12 lines)
    # Hint 1: Usa persim.wasserstein(dgm1, dgm2) para calcular la distancia
    # Hint 2: Asegúrate de manejar diagramas vacíos
    # Hint 3: Filtra puntos en la diagonal (birth == death)

    # YOUR CODE ENDS HERE''',
            'test_code': '''# Test automático
from notebooks.tda_tests import test_compute_wasserstein_distance
test_compute_wasserstein_distance(compute_wasserstein_distance)'''
        },
        {
            'num': 6,
            'name': 'detect_temporal_changes',
            'title': 'Detectar Cambios Temporales en Topología',
            'description': '''El análisis de **cambios temporales** en características topológicas permite identificar transiciones de estado cerebral, como el inicio de una crisis epiléptica o cambios en niveles de consciencia.

**Aplicación:** Detección temprana de eventos críticos en monitoreo de UCI o análisis de sueño.

**Tu tarea:** Implementa una función que detecte cambios significativos en la topología a lo largo del tiempo usando ventanas deslizantes.''',
            'instructions': '''    """
    Detecta cambios significativos en topología a lo largo del tiempo.

    Parámetros:
    -----------
    signal : array, shape (n_samples,)
        Señal temporal (ej: EEG)
    window_size : int
        Tamaño de ventana en muestras
    threshold : float
        Umbral de cambio (distancia de Wasserstein)

    Retorna:
    --------
    change_points : list
        Índices donde se detectaron cambios significativos
    distances : array
        Distancias entre ventanas consecutivas
    """
    # YOUR CODE STARTS HERE
    # (approx. 15-20 lines)
    # Hint 1: Divide señal en ventanas con sliding window
    # Hint 2: Calcula diagrama de persistencia para cada ventana
    # Hint 3: Compara ventanas consecutivas con Wasserstein distance
    # Hint 4: Marca como change_point donde distance > threshold

    # YOUR CODE ENDS HERE''',
            'test_code': '''# Test automático
from notebooks.tda_tests import test_detect_temporal_changes
test_detect_temporal_changes(detect_temporal_changes)'''
        },
        {
            'num': 7,
            'name': 'classify_spike_patterns',
            'title': 'Clasificar Patrones de Spikes con TDA',
            'description': '''Los **patrones de spikes neuronales** contienen información sobre el estado funcional del cerebro. Usando TDA podemos extraer características topológicas discriminativas para clasificar automáticamente diferentes tipos de actividad.

**Aplicación:** Clasificación de estados cerebrales (normal, preictal, ictal) en epilepsia o detección de patrones patológicos.

**Tu tarea:** Implementa un clasificador completo que use características TDA para clasificar patrones de spike trains.''',
            'instructions': '''    """
    Clasifica patrones de spike trains usando características TDA.

    Parámetros:
    -----------
    spike_trains_list : list of arrays
        Lista de spike trains a clasificar
    labels : array
        Etiquetas verdaderas (para entrenamiento)
    test_size : float
        Proporción de datos para test (default: 0.3)

    Retorna:
    --------
    classifier : objeto
        Clasificador entrenado
    accuracy : float
        Accuracy en conjunto de test
    predictions : array
        Predicciones en conjunto de test
    """
    # YOUR CODE STARTS HERE
    # (approx. 20-25 lines)
    # Hint 1: Extrae features TDA de cada spike train (Betti, persistence, entropy)
    # Hint 2: Crea matriz de features (n_samples, n_features)
    # Hint 3: Train/test split con sklearn.model_selection.train_test_split
    # Hint 4: Normaliza features con StandardScaler
    # Hint 5: Entrena RandomForestClassifier o SVC
    # Hint 6: Calcula accuracy en test set

    # YOUR CODE ENDS HERE''',
            'test_code': '''# Test automático
from notebooks.tda_tests import test_classify_spike_patterns
test_classify_spike_patterns(classify_spike_patterns)'''
        }
    ]

    # Encontrar índice donde insertar (después del último ejercicio)
    insert_idx = None
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == 'markdown' and '## 🎯 Resumen' in ''.join(cell.source):
            insert_idx = i
            break

    if insert_idx is None:
        # Si no encuentra resumen, insertar al final
        insert_idx = len(nb.cells)

    # Crear celdas para cada ejercicio
    new_cells = []

    for ex in new_exercises:
        # Celda markdown con título y descripción
        md_cell = nbf.v4.new_markdown_cell(f'''---

### Ejercicio {ex['num']} - {ex['name']}

{ex['description']}

**Dificultad:** ⭐⭐⭐ Avanzado
**Tiempo estimado:** 15-20 minutos''')
        new_cells.append(md_cell)

        # Celda de código con esqueleto
        code_cell = nbf.v4.new_code_cell(f'''def {ex['name']}():
{ex['instructions']}''')
        new_cells.append(code_cell)

        # Celda de test
        test_cell = nbf.v4.new_code_cell(ex['test_code'])
        new_cells.append(test_cell)

    # Insertar nuevas celdas
    for i, cell in enumerate(new_cells):
        nb.cells.insert(insert_idx + i, cell)

    # Guardar notebook actualizado
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)

    print(f"✅ Tutorial 2 expandido: 7 ejercicios totales (agregados 3 nuevos)")
    print(f"   • Ejercicio 5: compute_wasserstein_distance")
    print(f"   • Ejercicio 6: detect_temporal_changes")
    print(f"   • Ejercicio 7: classify_spike_patterns")

if __name__ == '__main__':
    try:
        expand_tutorial2()
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
