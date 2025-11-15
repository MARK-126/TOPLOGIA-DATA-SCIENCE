#!/usr/bin/env python3
"""
Script para expandir Tutorial 6 con 2 ejercicios adicionales avanzados
"""

import nbformat as nbf
import sys

def expand_tutorial6():
    """Expande Tutorial 6 con ejercicios 4, 5"""

    # Leer notebook existente
    notebook_path = 'notebooks/06_Caso_Estudio_Epilepsia_v2.ipynb'
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbf.read(f, as_version=4)

    # Nuevos ejercicios
    new_exercises = [
        {
            'num': 4,
            'name': 'feature_importance_analysis',
            'title': 'Análisis de Importancia de Features Topológicas',
            'description': '''Entender qué **features topológicas** son más discriminativas para detectar epilepsia permite interpretabilidad clínica y optimización del modelo. Features como Betti numbers específicos o rangos de persistencia pueden tener significado neurobiológico.

**Aplicación clínica:** Identificar biomarcadores interpretables, optimizar panel de features.

**Tu tarea:** Implementa un análisis completo de importancia de features usando múltiples métodos.''',
            'instructions': '''    """
    Analiza importancia de features topológicas para clasificación.

    Parámetros:
    -----------
    X : array, shape (n_samples, n_features)
        Matriz de features TDA
    y : array, shape (n_samples,)
        Labels (0=normal, 1=ictal)
    feature_names : list
        Nombres de las features

    Retorna:
    --------
    importance_scores : dict
        Diccionario con scores de diferentes métodos:
        - 'random_forest': Importancia de Random Forest
        - 'permutation': Importancia por permutación
        - 'mutual_info': Información mutua
    top_features : list
        Top 10 features más importantes (nombres)
    """
    # YOUR CODE STARTS HERE
    # (approx. 20-25 lines)
    # Hint 1: Entrena RandomForestClassifier y extrae feature_importances_
    # Hint 2: Calcula permutation importance con sklearn.inspection
    # Hint 3: Calcula mutual information con sklearn.feature_selection
    # Hint 4: Normaliza todos los scores a [0, 1]
    # Hint 5: Combina scores (promedio o ranking fusion)
    # Hint 6: Retorna top 10 features por importancia combinada

    # YOUR CODE ENDS HERE''',
            'test_code': '''# Test automático
from notebooks.tda_tests import test_feature_importance_analysis
test_feature_importance_analysis(feature_importance_analysis)'''
        },
        {
            'num': 5,
            'name': 'cross_validate_pipeline',
            'title': 'Validación Cruzada del Pipeline Completo',
            'description': '''La **validación cruzada rigurosa** es esencial para estimar el desempeño real del sistema en datos nuevos. Usar estratificación y validación temporal evita sobreajuste y produce estimaciones realistas.

**Aplicación:** Validación pre-clínica antes de deployment, reporte de desempeño confiable.

**Tu tarea:** Implementa un pipeline completo de validación cruzada con múltiples métricas.''',
            'instructions': '''    """
    Valida pipeline completo de detección usando cross-validation.

    Parámetros:
    -----------
    eeg_data : array, shape (n_epochs, n_channels, n_samples)
        Datos EEG crudos
    labels : array, shape (n_epochs,)
        Labels verdaderas
    cv_folds : int
        Número de folds para cross-validation (default: 5)

    Retorna:
    --------
    cv_results : dict
        Resultados de validación cruzada:
        - 'accuracy': Accuracy promedio ± std
        - 'precision': Precision promedio ± std
        - 'recall': Recall promedio ± std
        - 'f1': F1-score promedio ± std
        - 'roc_auc': ROC-AUC promedio ± std
        - 'confusion_matrices': Lista de matrices de confusión
    trained_model : objeto
        Modelo final entrenado en todos los datos
    """
    # YOUR CODE STARTS HERE
    # (approx. 25-30 lines)
    # Hint 1: Preprocesa todos los epochs de EEG (filtros, CAR, normalización)
    # Hint 2: Extrae features TDA + espectrales de cada epoch
    # Hint 3: Usa StratifiedKFold para splits balanceados
    # Hint 4: Para cada fold:
    #         - Entrena pipeline en train set
    #         - Evalúa en validation set
    #         - Calcula todas las métricas
    # Hint 5: Agrega resultados y calcula media ± std
    # Hint 6: Entrena modelo final en todos los datos
    # Hint 7: Retorna resultados completos de CV

    # YOUR CODE ENDS HERE''',
            'test_code': '''# Test automático
from notebooks.tda_tests import test_cross_validate_pipeline
test_cross_validate_pipeline(cross_validate_pipeline)'''
        }
    ]

    # Encontrar índice
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

    print(f"✅ Tutorial 6 expandido: 5 ejercicios totales (agregados 2 nuevos)")
    print(f"   • Ejercicio 4: feature_importance_analysis")
    print(f"   • Ejercicio 5: cross_validate_pipeline")

if __name__ == '__main__':
    try:
        expand_tutorial6()
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
