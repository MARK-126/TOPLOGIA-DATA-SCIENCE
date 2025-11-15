#!/usr/bin/env python3
"""
Script para expandir Tutorial 5 con 3 ejercicios adicionales avanzados
"""

import nbformat as nbf
import sys

def expand_tutorial5():
    """Expande Tutorial 5 con ejercicios 4, 5, 6"""

    # Leer notebook existente
    notebook_path = 'notebooks/05_Series_Temporales_EEG_v2.ipynb'
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbf.read(f, as_version=4)

    # Nuevos ejercicios
    new_exercises = [
        {
            'num': 4,
            'name': 'compute_delay_embedding_dim',
            'title': 'Calcular Dimensión de Embedding Óptima',
            'description': '''La **dimensión de embedding** óptima es crucial para reconstruir correctamente el espacio de estados de un sistema dinámico. El método de **False Nearest Neighbors (FNN)** determina la dimensión mínima que desenrolla completamente el atractor.

**Aplicación:** Análisis de series temporales EEG para caracterizar dinámicas cerebrales.

**Tu tarea:** Implementa el algoritmo de False Nearest Neighbors para determinar la dimensión de embedding óptima.''',
            'instructions': '''    """
    Calcula dimensión de embedding óptima usando False Nearest Neighbors.

    Parámetros:
    -----------
    signal : array, shape (n_samples,)
        Serie temporal
    delay : int
        Delay para embedding (tau)
    max_dim : int
        Dimensión máxima a probar (default: 10)
    rtol : float
        Tolerancia relativa para FNN (default: 10.0)

    Retorna:
    --------
    optimal_dim : int
        Dimensión de embedding óptima
    fnn_percentages : array
        Porcentaje de FNN para cada dimensión
    """
    # YOUR CODE STARTS HERE
    # (approx. 18-25 lines)
    # Hint 1: Para cada dimensión d de 1 a max_dim:
    # Hint 2:   Construye embedding de dimensión d y d+1
    # Hint 3:   Para cada punto en embedding d, encuentra vecino más cercano
    # Hint 4:   Calcula distancia en embedding d+1 para mismo par
    # Hint 5:   Si distancia crece > rtol, marca como False Nearest Neighbor
    # Hint 6: Dimensión óptima = donde %FNN < 1%

    # YOUR CODE ENDS HERE''',
            'test_code': '''# Test automático
from notebooks.tda_tests import test_compute_delay_embedding_dim
test_compute_delay_embedding_dim(compute_delay_embedding_dim)'''
        },
        {
            'num': 5,
            'name': 'reconstruct_attractor',
            'title': 'Reconstruir Atractor desde Serie Temporal',
            'description': '''La **reconstrucción de atractores** permite visualizar y analizar la dinámica del sistema cerebral en un espacio de estados. Combinando embedding de Takens con análisis topológico se puede caracterizar la complejidad del atractor.

**Aplicación:** Visualización de dinámicas cerebrales, comparación de atractores entre condiciones.

**Tu tarea:** Implementa una función completa de reconstrucción y caracterización de atractores.''',
            'instructions': '''    """
    Reconstruye y caracteriza el atractor de una serie temporal.

    Parámetros:
    -----------
    signal : array, shape (n_samples,)
        Serie temporal EEG
    delay : int, optional
        Delay (si None, se calcula automáticamente)
    embedding_dim : int, optional
        Dimensión de embedding (si None, se calcula con FNN)

    Retorna:
    --------
    attractor : array, shape (n_points, embedding_dim)
        Atractor reconstruido
    characteristics : dict
        - 'correlation_dimension': Dimensión de correlación
        - 'lyapunov_exponent': Exponente de Lyapunov aproximado
        - 'betti_numbers': Números de Betti del atractor
        - 'persistence_entropy': Entropía de persistencia
    """
    # YOUR CODE STARTS HERE
    # (approx. 22-30 lines)
    # Hint 1: Si delay es None, calcula con autocorrelación
    # Hint 2: Si embedding_dim es None, calcula con FNN
    # Hint 3: Construye embedding de Takens
    # Hint 4: Calcula correlation dimension (Grassberger-Procaccia)
    # Hint 5: Estima Lyapunov exponent (mayor exponente)
    # Hint 6: Calcula homología persistente del atractor
    # Hint 7: Extrae Betti numbers y persistence entropy

    # YOUR CODE ENDS HERE''',
            'test_code': '''# Test automático
from notebooks.tda_tests import test_reconstruct_attractor
test_reconstruct_attractor(reconstruct_attractor)'''
        },
        {
            'num': 6,
            'name': 'predict_next_event',
            'title': 'Predecir Próximo Evento usando TDA',
            'description': '''La **predicción de eventos** (como crisis epilépticas) usando características topológicas permite alertas tempranas. Al detectar cambios topológicos precursores, podemos predecir eventos antes de que ocurran.

**Aplicación clínica:** Sistema de alerta temprana de crisis epilépticas, predicción de arritmias cardíacas.

**Tu tarea:** Implementa un predictor de eventos basado en cambios topológicos en ventanas temporales.''',
            'instructions': '''    """
    Predice próximo evento crítico usando análisis topológico temporal.

    Parámetros:
    -----------
    signal : array, shape (n_samples,)
        Serie temporal continua
    event_labels : array, shape (n_samples,)
        Labels de eventos (0=normal, 1=evento)
    window_size : int
        Tamaño de ventana para análisis (default: 256)
    prediction_horizon : int
        Horizontes de predicción en muestras (default: 128)

    Retorna:
    --------
    predictions : array
        Predicciones binarias (0=no evento, 1=evento próximo)
    probabilities : array
        Probabilidades de evento
    roc_auc : float
        AUC de la curva ROC
    """
    # YOUR CODE STARTS HERE
    # (approx. 25-35 lines)
    # Hint 1: Divide señal en ventanas con sliding window
    # Hint 2: Para cada ventana, extrae features TDA (Betti, persistence, entropy)
    # Hint 3: Crea labels desplazadas por prediction_horizon
    # Hint 4: Entrena clasificador (RandomForest o GradientBoosting)
    # Hint 5: Predice probabilidad de evento en próximo horizonte
    # Hint 6: Calcula ROC-AUC para evaluar desempeño

    # YOUR CODE ENDS HERE''',
            'test_code': '''# Test automático
from notebooks.tda_tests import test_predict_next_event
test_predict_next_event(predict_next_event)'''
        }
    ]

    # Encontrar índice de inserción
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
**Tiempo estimado:** 20-25 minutos''')
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

    print(f"✅ Tutorial 5 expandido: 6 ejercicios totales (agregados 3 nuevos)")
    print(f"   • Ejercicio 4: compute_delay_embedding_dim")
    print(f"   • Ejercicio 5: reconstruct_attractor")
    print(f"   • Ejercicio 6: predict_next_event")

if __name__ == '__main__':
    try:
        expand_tutorial5()
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
