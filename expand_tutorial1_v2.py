#!/usr/bin/env python3
"""
Script para expandir Tutorial 1 v2 con ejercicios adicionales
"""

import nbformat as nbf

# Leer notebook existente
with open('notebooks/01_Introduccion_TDA_v2.ipynb', 'r') as f:
    nb = nbf.read(f, as_version=4)

# Encontrar posición para insertar nuevos ejercicios (antes del resumen final)
# Los nuevos ejercicios irán entre el Ejercicio 4 y el Resumen
insert_idx = len(nb.cells) - 1  # Antes de la última celda (resumen)

new_cells = []

# ========== EJERCICIO 5: Comparar características topológicas ==========
new_cells.append(nbf.v4.new_markdown_cell("""### Ejercicio 5 - compare_topological_features

Compara características topológicas entre dos datasets.

**Objetivo:** Cuantificar similitud topológica entre estados cerebrales

**Instrucciones:**
1. Calcular homología persistente para ambos datasets
2. Extraer características: max persistence, total persistence, n_cycles
3. Calcular distancia euclidiana entre vectores de características
4. Retornar diccionario con características y distancia"""))

new_cells.append(nbf.v4.new_code_cell("""# EJERCICIO 5: Comparar Características Topológicas

def compare_topological_features(data1, data2, max_dim=2):
    \"\"\"
    Compara características topológicas entre dos datasets.

    Arguments:
    data1 -- primer dataset (n_samples1, n_features)
    data2 -- segundo dataset (n_samples2, n_features)
    max_dim -- dimensión máxima para homología

    Returns:
    features1 -- diccionario con características del dataset 1
    features2 -- diccionario con características del dataset 2
    distance -- distancia euclidiana entre vectores de características
    \"\"\"

    # 1. Calcular homología para data1
    # (approx. 2 lines)
    # YOUR CODE STARTS HERE


    # YOUR CODE ENDS HERE

    # 2. Calcular homología para data2
    # (approx. 2 lines)
    # YOUR CODE STARTS HERE


    # YOUR CODE ENDS HERE

    # 3. Extraer características de H1 (ciclos) para data1
    # (approx. 8 lines)
    # Características: n_cycles, max_persistence, total_persistence
    # YOUR CODE STARTS HERE








    # YOUR CODE ENDS HERE

    # 4. Extraer características de H1 para data2
    # (approx. 8 lines)
    # YOUR CODE STARTS HERE








    # YOUR CODE ENDS HERE

    # 5. Calcular distancia euclidiana entre vectores de características
    # (approx. 4 lines)
    # Crear vectores [n_cycles, max_persistence, total_persistence]
    # YOUR CODE STARTS HERE




    # YOUR CODE ENDS HERE

    return features1, features2, distance"""))

new_cells.append(nbf.v4.new_code_cell("""# Test del Ejercicio 5
f1, f2, dist = compare_topological_features(resting_state, active_state, max_dim=2)

print("Características topológicas:")
print(f"\\nEstado de reposo:")
print(f"  • Ciclos (H₁): {f1['n_cycles']}")
print(f"  • Max persistencia: {f1['max_persistence']:.3f}")
print(f"  • Persistencia total: {f1['total_persistence']:.3f}")

print(f"\\nEstado activo:")
print(f"  • Ciclos (H₁): {f2['n_cycles']}")
print(f"  • Max persistencia: {f2['max_persistence']:.3f}")
print(f"  • Persistencia total: {f2['total_persistence']:.3f}")

print(f"\\n📊 Distancia topológica: {dist:.3f}")
print("(Mayor distancia → estados más diferentes topológicamente)")

# Test automático
from tda_tests import test_compare_topological_features
test_compare_topological_features(compare_topological_features)"""))

new_cells.append(nbf.v4.new_markdown_cell("""<div style="background-color:#e3f2fd; padding:15px; border-left:5px solid #2196f3; margin: 20px 0;">

**💡 Interpretación:**

- **Distancia alta** → Estados cerebrales topológicamente diferentes
- **Distancia baja** → Estados similares (mismo nivel de organización)
- Útil para **clasificación de estados** cognitivos
- Robusto al ruido en comparación con métricas tradicionales

</div>

---"""))

# ========== EJERCICIO 6: Filtrar por persistencia ==========
new_cells.append(nbf.v4.new_markdown_cell("""### Ejercicio 6 - filter_by_persistence

Filtra características topológicas por su persistencia.

**Objetivo:** Eliminar ruido y mantener solo características significativas

**Concepto:** La persistencia (death - birth) mide cuán "robusta" es una característica.
Características con baja persistencia suelen ser ruido.

**Instrucciones:**
1. Calcular persistencia para cada característica
2. Filtrar características con persistencia >= threshold
3. Retornar diagrama filtrado"""))

new_cells.append(nbf.v4.new_code_cell("""# EJERCICIO 6: Filtrar por Persistencia

def filter_by_persistence(persistence_diagram, threshold=0.1):
    \"\"\"
    Filtra características topológicas por persistencia mínima.

    Arguments:
    persistence_diagram -- array (n_features, 2) con (birth, death)
    threshold -- persistencia mínima para mantener característica

    Returns:
    filtered_diagram -- diagrama filtrado
    n_removed -- número de características removidas
    \"\"\"

    # 1. Calcular persistencia para cada característica
    # (approx. 1 line)
    # Persistencia = death - birth
    # YOUR CODE STARTS HERE

    # YOUR CODE ENDS HERE

    # 2. Identificar características significativas
    # (approx. 2 lines)
    # Mantener solo donde persistencia >= threshold Y death es finito
    # YOUR CODE STARTS HERE


    # YOUR CODE ENDS HERE

    # 3. Filtrar diagrama
    # (approx. 1 line)
    # YOUR CODE STARTS HERE

    # YOUR CODE ENDS HERE

    # 4. Contar cuántas se removieron
    # (approx. 1 line)
    # YOUR CODE STARTS HERE

    # YOUR CODE ENDS HERE

    return filtered_diagram, n_removed"""))

new_cells.append(nbf.v4.new_code_cell("""# Test del Ejercicio 6
# Generar diagrama de prueba
result = ripser(circle_points, maxdim=1)
dgm_h1 = result['dgms'][1]

print(f"Diagrama original (H₁): {len(dgm_h1)} características")

# Filtrar con diferentes thresholds
filtered_01, n_removed_01 = filter_by_persistence(dgm_h1, threshold=0.1)
filtered_02, n_removed_02 = filter_by_persistence(dgm_h1, threshold=0.2)

print(f"\\nCon threshold 0.1: {len(filtered_01)} características ({n_removed_01} removidas)")
print(f"Con threshold 0.2: {len(filtered_02)} características ({n_removed_02} removidas)")

# Visualizar
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Original
from persim import plot_diagrams
plot_diagrams([dgm_h1], ax=axes[0])
axes[0].set_title('Diagrama Original', fontsize=12, fontweight='bold')

# Filtrado
plot_diagrams([filtered_02], ax=axes[1])
axes[1].set_title(f'Diagrama Filtrado (threshold=0.2)\\n{len(filtered_02)} características persistentes',
                 fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()

# Test automático
from tda_tests import test_filter_by_persistence
test_filter_by_persistence(filter_by_persistence)"""))

new_cells.append(nbf.v4.new_markdown_cell("""<div style="background-color:#fff3cd; padding:15px; border-left:5px solid #ffc107; margin: 20px 0;">

**⚙️ Uso Práctico:**

- **Preprocesamiento:** Eliminar ruido antes de análisis
- **Visualización:** Diagramas más limpios y legibles
- **Machine Learning:** Features más robustas
- **Regla general:** threshold = 10-20% del rango de distancias

</div>

---"""))

# ========== EJERCICIO 7: Entropía de persistencia ==========
new_cells.append(nbf.v4.new_markdown_cell("""### Ejercicio 7 - compute_persistence_entropy

Calcula la entropía de persistencia como medida de complejidad.

**Concepto:** La entropía mide cuán uniforme es la distribución de persistencias.
- **Alta entropía:** Muchas características con persistencias similares
- **Baja entropía:** Pocas características dominan

**Fórmula:** $E = -\\sum p_i \\log(p_i)$ donde $p_i = \\frac{\\text{persistence}_i}{\\sum \\text{persistence}_j}$

**Aplicación:** Cuantificar complejidad estructural de estados cerebrales"""))

new_cells.append(nbf.v4.new_code_cell("""# EJERCICIO 7: Entropía de Persistencia

def compute_persistence_entropy(persistence_diagram):
    \"\"\"
    Calcula entropía de persistencia como medida de complejidad.

    Arguments:
    persistence_diagram -- array (n_features, 2) con (birth, death)

    Returns:
    entropy -- entropía de persistencia
    \"\"\"

    # 1. Filtrar características infinitas
    # (approx. 2 lines)
    # Mantener solo donde death es finito
    # YOUR CODE STARTS HERE


    # YOUR CODE ENDS HERE

    # 2. Calcular persistencias
    # (approx. 1 line)
    # YOUR CODE STARTS HERE

    # YOUR CODE ENDS HERE

    # 3. Normalizar a probabilidades
    # (approx. 2 lines)
    # p_i = persistence_i / sum(persistences)
    # YOUR CODE STARTS HERE


    # YOUR CODE ENDS HERE

    # 4. Calcular entropía
    # (approx. 3 lines)
    # E = -sum(p * log(p)) donde p > 0
    # YOUR CODE STARTS HERE



    # YOUR CODE ENDS HERE

    return entropy"""))

new_cells.append(nbf.v4.new_code_cell("""# Test del Ejercicio 7
# Comparar entropía de diferentes estados
result_resting = ripser(resting_state, maxdim=1)
result_active = ripser(active_state, maxdim=1)

entropy_resting_h1 = compute_persistence_entropy(result_resting['dgms'][1])
entropy_active_h1 = compute_persistence_entropy(result_active['dgms'][1])

print("Entropía de Persistencia (H₁):")
print(f"\\nEstado de reposo: {entropy_resting_h1:.3f}")
print(f"Estado activo:    {entropy_active_h1:.3f}")

diff = abs(entropy_resting_h1 - entropy_active_h1)
print(f"\\nDiferencia: {diff:.3f}")

if entropy_resting_h1 > entropy_active_h1:
    print("→ Estado de reposo tiene mayor complejidad topológica (H₁)")
else:
    print("→ Estado activo tiene mayor complejidad topológica (H₁)")

# Calcular también para H₂
if len(result_resting['dgms']) > 2:
    entropy_resting_h2 = compute_persistence_entropy(result_resting['dgms'][2])
    entropy_active_h2 = compute_persistence_entropy(result_active['dgms'][2])
    print(f"\\nEntropía H₂ (cavidades):")
    print(f"  Reposo: {entropy_resting_h2:.3f}")
    print(f"  Activo: {entropy_active_h2:.3f}")

# Test automático
from tda_tests import test_compute_persistence_entropy
test_compute_persistence_entropy(compute_persistence_entropy)"""))

new_cells.append(nbf.v4.new_markdown_cell("""<div style="background-color:#e8f5e9; padding:15px; border-left:5px solid #4caf50; margin: 20px 0;">

**💡 Interpretación Clínica:**

- **Alta entropía** → Complejidad distribuida (muchas estructuras similares)
- **Baja entropía** → Pocas estructuras dominantes
- Útil para clasificar **trastornos neurológicos**:
  - Alzheimer: Reducción en entropía H₁ (pérdida de ciclos funcionales)
  - Esquizofrenia: Alteración en entropía H₂ (organización jerárquica)
- Puede usarse como **biomarcador diagnóstico**

</div>

---"""))

# Actualizar tabla de contenidos
for i, cell in enumerate(nb.cells):
    if cell.cell_type == 'markdown' and '<a name=\'toc\'></a>' in cell.source:
        # Actualizar TOC
        nb.cells[i].source = """<a name='toc'></a>
## 📚 Tabla de Contenidos

- [1 - Setup e Importaciones](#1)
- [2 - Conceptos Fundamentales de Topología](#2)
- [3 - Complejos Simpliciales](#3)
    - [Ejercicio 1 - build_simplicial_complex](#ex-1)
- [4 - Números de Betti y Homología](#4)
    - [Ejercicio 2 - compute_betti_numbers](#ex-2)
- [5 - Aplicación: Redes Neuronales](#5)
    - [Ejercicio 3 - generate_neural_network](#ex-3)
- [6 - Aplicación: Estados Cerebrales](#6)
    - [Ejercicio 4 - generate_brain_state](#ex-4)
- [6.5 - Ejercicios Avanzados](#6.5)
    - [Ejercicio 5 - compare_topological_features](#ex-5)
    - [Ejercicio 6 - filter_by_persistence](#ex-6)
    - [Ejercicio 7 - compute_persistence_entropy](#ex-7)
- [7 - Resumen y Próximos Pasos](#7)

---"""
        break

# Insertar sección de ejercicios avanzados
header_cell = nbf.v4.new_markdown_cell("""<a name='6.5'></a>
## 6.5 - Ejercicios Avanzados: Análisis Topológico Profundo

[Volver al índice](#toc)

Ahora aplicaremos técnicas más avanzadas para comparar y analizar características topológicas.

---""")

# Insertar nuevas celdas antes del resumen
nb.cells.insert(insert_idx, header_cell)
for i, cell in enumerate(new_cells):
    nb.cells.insert(insert_idx + 1 + i, cell)

# Guardar notebook expandido
with open('notebooks/01_Introduccion_TDA_v2.ipynb', 'w') as f:
    nbf.write(nb, f)

print("✅ Tutorial 1 expandido: 7 ejercicios totales (agregados 3 nuevos)")
print("   • Ejercicio 5: compare_topological_features")
print("   • Ejercicio 6: filter_by_persistence")
print("   • Ejercicio 7: compute_persistence_entropy")
