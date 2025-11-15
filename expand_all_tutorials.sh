#!/bin/bash

echo "🚀 Expandiendo todos los tutoriales con ejercicios adicionales..."
echo ""

cd /home/user/TOPLOGIA-DATA-SCIENCE

# Ya expandimos Tutorial 1
echo "✅ Tutorial 1: Ya expandido (7 ejercicios)"

# Crear scripts para los demás
echo "📝 Creando scripts de expansión..."

# Los scripts se crearán en Python
python3 << 'EOF'
import nbformat as nbf

# =========================================================================
# TUTORIAL 2: Agregar 3 ejercicios nuevos
# =========================================================================
print("\n🔧 Expandiendo Tutorial 2...")

with open('notebooks/02_Homologia_Persistente_Avanzada_v2.ipynb', 'r') as f:
    nb2 = nbf.read(f, as_version=4)

# Ejercicios nuevos para Tutorial 2:
# 5. compute_wasserstein_distance - Distancia de Wasserstein entre diagramas
# 6. detect_temporal_changes - Detectar cambios en series temporales
# 7. classify_spike_patterns - Clasificar patrones de spikes con TDA

insert_idx2 = len(nb2.cells) - 1

new_cells2 = []

# Ejercicio 5
new_cells2.append(nbf.v4.new_markdown_cell("""### Ejercicio 5 - compute_wasserstein_distance

Calcula la distancia de Wasserstein entre dos diagramas de persistencia.

**Objetivo:** Cuantificar similitud entre patrones de spikes

**Concepto:** La distancia de Wasserstein mide el "costo" de transformar un diagrama en otro."""))

new_cells2.append(nbf.v4.new_code_cell("""# EJERCICIO 5: Distancia de Wasserstein

def compute_wasserstein_distance(dgm1, dgm2, order=2):
    \"\"\"
    Calcula distancia de Wasserstein entre dos diagramas.

    Arguments:
    dgm1, dgm2 -- diagramas de persistencia
    order -- orden de la distancia (1 o 2)

    Returns:
    distance -- distancia de Wasserstein
    \"\"\"
    from persim import sliced_wasserstein

    # 1. Filtrar puntos infinitos
    # (approx. 4 lines)
    # YOUR CODE STARTS HERE




    # YOUR CODE ENDS HERE

    # 2. Calcular distancia usando persim
    # (approx. 2 lines)
    # YOUR CODE STARTS HERE


    # YOUR CODE ENDS HERE

    return distance"""))

# Ejercicio 6  
new_cells2.append(nbf.v4.new_markdown_cell("""### Ejercicio 6 - detect_temporal_changes

Detecta cambios temporales en actividad neuronal usando persistencia deslizante.

**Aplicación:** Detección de crisis epilépticas, transiciones de estados"""))

new_cells2.append(nbf.v4.new_code_cell("""# EJERCICIO 6: Detección de Cambios Temporales

def detect_temporal_changes(spike_train, window_size=100, stride=50):
    \"\"\"
    Detecta cambios usando ventanas deslizantes de persistencia.

    Arguments:
    spike_train -- tiempos de spikes (array 1D)
    window_size -- tamaño de ventana
    stride -- paso entre ventanas

    Returns:
    time_points -- puntos temporales centrales
    persistence_values -- persistencia máxima en cada ventana
    change_points -- índices donde hay cambios significativos
    \"\"\"

    # 1. Ventanas deslizantes
    # (approx. 6 lines)
    # YOUR CODE STARTS HERE






    # YOUR CODE ENDS HERE

    # 2. Detectar cambios (gradiente alto)
    # (approx. 3 lines)
    # YOUR CODE STARTS HERE



    # YOUR CODE ENDS HERE

    return time_points, persistence_values, change_points"""))

# Ejercicio 7
new_cells2.append(nbf.v4.new_markdown_cell("""### Ejercicio 7 - classify_spike_patterns

Clasifica patrones de spikes usando características topológicas.

**Objetivo:** Distinguir entre tipos de actividad neuronal"""))

new_cells2.append(nbf.v4.new_code_cell("""# EJERCICIO 7: Clasificación de Patrones

def classify_spike_patterns(spike_trains_dict, test_size=0.3):
    \"\"\"
    Clasifica patrones usando Random Forest y features TDA.

    Arguments:
    spike_trains_dict -- dict {pattern_name: [spike_trains_list]}
    test_size -- proporción de test

    Returns:
    clf -- clasificador entrenado
    accuracy -- precisión en test
    \"\"\"
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    X_data = []
    y_data = []

    # 1. Extraer features TDA de cada spike train
    # (approx. 10 lines)
    # YOUR CODE STARTS HERE










    # YOUR CODE ENDS HERE

    # 2. Train/test split y clasificación
    # (approx. 5 lines)
    # YOUR CODE STARTS HERE





    # YOUR CODE ENDS HERE

    return clf, accuracy"""))

# Insertar nuevas celdas
header2 = nbf.v4.new_markdown_cell("""## Ejercicios Avanzados: Análisis Comparativo

---""")
nb2.cells.insert(insert_idx2, header2)
for i, cell in enumerate(new_cells2):
    nb2.cells.insert(insert_idx2 + 1 + i, cell)

with open('notebooks/02_Homologia_Persistente_Avanzada_v2.ipynb', 'w') as f:
    nbf.write(nb2, f)

print("✅ Tutorial 2 expandido: 7 ejercicios totales (agregados 3 nuevos)")

# =========================================================================
# TUTORIAL 3: Conectividad Cerebral - Agregar 3 ejercicios
# =========================================================================
print("\n🔧 Expandiendo Tutorial 3...")

with open('notebooks/03_Conectividad_Cerebral_v2.ipynb', 'r') as f:
    nb3 = nbf.read(f, as_version=4)

insert_idx3 = len(nb3.cells) - 1
new_cells3 = []

# Ejercicio 4
new_cells3.append(nbf.v4.new_markdown_cell("""### Ejercicio 4 - compute_graph_features

Calcula características de grafo complementarias a TDA.

**Incluir:** Clustering coefficient, betweenness centrality, modularity"""))

new_cells3.append(nbf.v4.new_code_cell("""# EJERCICIO 4: Características de Grafo

def compute_graph_features(conn_matrix, threshold=0.3):
    \"\"\"
    Calcula características de teoría de grafos.

    Arguments:
    conn_matrix -- matriz de conectividad
    threshold -- umbral para binarizar

    Returns:
    features -- dict con características del grafo
    \"\"\"
    import networkx as nx

    # 1. Crear grafo desde matriz de conectividad
    # (approx. 3 lines)
    # YOUR CODE STARTS HERE



    # YOUR CODE ENDS HERE

    # 2. Calcular métricas
    # (approx. 8 lines)
    # clustering, betweenness, degree distribution, etc.
    # YOUR CODE STARTS HERE








    # YOUR CODE ENDS HERE

    return features"""))

# Ejercicio 5
new_cells3.append(nbf.v4.new_markdown_cell("""### Ejercicio 5 - find_critical_nodes

Identifica nodos críticos usando persistencia local.

**Aplicación:** Encontrar regiones cerebrales esenciales"""))

new_cells3.append(nbf.v4.new_code_cell("""# EJERCICIO 5: Nodos Críticos

def find_critical_nodes(conn_matrix, top_k=5):
    \"\"\"
    Identifica nodos más críticos topológicamente.

    Arguments:
    conn_matrix -- matriz de conectividad
    top_k -- número de nodos críticos a retornar

    Returns:
    critical_nodes -- índices de nodos críticos
    criticality_scores -- scores de criticidad
    \"\"\"

    # 1. Para cada nodo, calcular su impacto en conectividad global
    # (approx. 10 lines)
    # Remover temporalmente cada nodo y medir cambio en β₀
    # YOUR CODE STARTS HERE










    # YOUR CODE ENDS HERE

    # 2. Ordenar y retornar top k
    # (approx. 3 lines)
    # YOUR CODE STARTS HERE



    # YOUR CODE ENDS HERE

    return critical_nodes, criticality_scores"""))

# Ejercicio 6
new_cells3.append(nbf.v4.new_markdown_cell("""### Ejercicio 6 - track_connectivity_evolution

Analiza cómo evoluciona la conectividad en el tiempo.

**Aplicación:** Plasticidad sináptica, aprendizaje"""))

new_cells3.append(nbf.v4.new_code_cell("""# EJERCICIO 6: Evolución de Conectividad

def track_connectivity_evolution(connectivity_matrices, time_points):
    \"\"\"
    Rastrea evolución temporal de características topológicas.

    Arguments:
    connectivity_matrices -- lista de matrices en diferentes tiempos
    time_points -- puntos temporales correspondientes

    Returns:
    evolution_features -- dict con series temporales de features
    \"\"\"

    # 1. Para cada matriz, extraer features topológicas
    # (approx. 12 lines)
    # YOUR CODE STARTS HERE












    # YOUR CODE ENDS HERE

    return evolution_features"""))

header3 = nbf.v4.new_markdown_cell("""## Ejercicios Avanzados: Análisis de Redes

---""")
nb3.cells.insert(insert_idx3, header3)
for i, cell in enumerate(new_cells3):
    nb3.cells.insert(insert_idx3 + 1 + i, cell)

with open('notebooks/03_Conectividad_Cerebral_v2.ipynb', 'w') as f:
    nbf.write(nb3, f)

print("✅ Tutorial 3 expandido: 6 ejercicios totales (agregados 3 nuevos)")

# =========================================================================
# TUTORIAL 4: Mapper - Agregar 2 ejercicios
# =========================================================================
print("\n🔧 Expandiendo Tutorial 4...")

with open('notebooks/04_Mapper_Algorithm_v2.ipynb', 'r') as f:
    nb4 = nbf.read(f, as_version=4)

insert_idx4 = len(nb4.cells) - 1
new_cells4 = []

# Ejercicio 4
new_cells4.append(nbf.v4.new_markdown_cell("""### Ejercicio 4 - optimize_mapper_parameters

Optimiza parámetros de Mapper para mejor visualización.

**Objetivo:** Encontrar n_intervals y overlap óptimos"""))

new_cells4.append(nbf.v4.new_code_cell("""# EJERCICIO 4: Optimización de Parámetros

def optimize_mapper_parameters(data, filter_values, param_grid):
    \"\"\"
    Busca mejores parámetros de Mapper.

    Arguments:
    data -- datos originales
    filter_values -- valores del filtro
    param_grid -- dict con rangos {'n_intervals': [...], 'overlap': [...]}

    Returns:
    best_params -- mejores parámetros encontrados
    best_score -- score de calidad
    \"\"\"

    # 1. Grid search sobre parámetros
    # (approx. 12 lines)
    # Métrica: maximizar número de nodos sin fragmentar
    # YOUR CODE STARTS HERE












    # YOUR CODE ENDS HERE

    return best_params, best_score"""))

# Ejercicio 5  
new_cells4.append(nbf.v4.new_markdown_cell("""### Ejercicio 5 - detect_loops_in_mapper

Detecta loops (ciclos) en el grafo de Mapper.

**Aplicación:** Identificar procesos recurrentes en dinámicas cerebrales"""))

new_cells4.append(nbf.v4.new_code_cell("""# EJERCICIO 5: Detección de Loops

def detect_loops_in_mapper(G):
    \"\"\"
    Detecta ciclos en el grafo de Mapper.

    Arguments:
    G -- grafo de NetworkX

    Returns:
    loops -- lista de ciclos encontrados
    loop_lengths -- longitudes de cada ciclo
    \"\"\"
    import networkx as nx

    # 1. Encontrar todos los ciclos simples
    # (approx. 4 lines)
    # YOUR CODE STARTS HERE




    # YOUR CODE ENDS HERE

    # 2. Clasificar por longitud
    # (approx. 3 lines)
    # YOUR CODE STARTS HERE



    # YOUR CODE ENDS HERE

    return loops, loop_lengths"""))

header4 = nbf.v4.new_markdown_cell("""## Ejercicios Avanzados: Optimización

---""")
nb4.cells.insert(insert_idx4, header4)
for i, cell in enumerate(new_cells4):
    nb4.cells.insert(insert_idx4 + 1 + i, cell)

with open('notebooks/04_Mapper_Algorithm_v2.ipynb', 'w') as f:
    nbf.write(nb4, f)

print("✅ Tutorial 4 expandido: 5 ejercicios totales (agregados 2 nuevos)")

# =========================================================================
# TUTORIAL 5: Series Temporales - Agregar 3 ejercicios
# =========================================================================
print("\n🔧 Expandiendo Tutorial 5...")

with open('notebooks/05_Series_Temporales_EEG_v2.ipynb', 'r') as f:
    nb5 = nbf.read(f, as_version=4)

insert_idx5 = len(nb5.cells) - 1
new_cells5 = []

# Ejercicio 4
new_cells5.append(nbf.v4.new_markdown_cell("""### Ejercicio 4 - compute_delay_embedding_dim

Estima dimensión óptima de embedding usando False Nearest Neighbors.

**Objetivo:** Determinar dim óptima para reconstrucción"""))

new_cells5.append(nbf.v4.new_code_cell("""# EJERCICIO 4: Dimensión de Embedding Óptima

def compute_delay_embedding_dim(signal, delay, max_dim=10):
    \"\"\"
    Estima dimensión óptima de embedding.

    Arguments:
    signal -- señal temporal 1D
    delay -- delay a usar
    max_dim -- dimensión máxima a probar

    Returns:
    optimal_dim -- dimensión óptima estimada
    fnn_percentages -- porcentajes de FNN para cada dim
    \"\"\"

    # 1. Método de False Nearest Neighbors
    # (approx. 15 lines)
    # Para cada dim, contar % de vecinos "falsos"
    # YOUR CODE STARTS HERE















    # YOUR CODE ENDS HERE

    return optimal_dim, fnn_percentages"""))

# Ejercicio 5
new_cells5.append(nbf.v4.new_markdown_cell("""### Ejercicio 5 - reconstruct_attractor

Reconstruye el atractor desde una serie temporal.

**Aplicación:** Análisis de dinámicas cerebrales complejas"""))

new_cells5.append(nbf.v4.new_code_cell("""# EJERCICIO 5: Reconstrucción de Atractor

def reconstruct_attractor(signal, delay=None, dimension=3):
    \"\"\"
    Reconstruye atractor con Takens + analiza topología.

    Arguments:
    signal -- señal temporal
    delay -- delay (None = auto)
    dimension -- dimensión de embedding

    Returns:
    attractor -- puntos del atractor reconstruido
    topological_features -- características topológicas
    \"\"\"

    # 1. Crear embedding
    # (approx. 3 lines)
    # YOUR CODE STARTS HERE



    # YOUR CODE ENDS HERE

    # 2. Analizar topología del atractor
    # (approx. 8 lines)
    # Calcular persistencia y extraer features
    # YOUR CODE STARTS HERE








    # YOUR CODE ENDS HERE

    return attractor, topological_features"""))

# Ejercicio 6
new_cells5.append(nbf.v4.new_markdown_cell("""### Ejercicio 6 - predict_next_event

Predice próximo evento usando features topológicas.

**Aplicación:** Predicción de crisis epilépticas"""))

new_cells5.append(nbf.v4.new_code_cell("""# EJERCICIO 6: Predicción de Eventos

def predict_next_event(signal, window_size=500, horizon=100):
    \"\"\"
    Predice si habrá evento en próximo horizonte.

    Arguments:
    signal -- señal temporal completa
    window_size -- tamaño de ventana de análisis
    horizon -- horizonte de predicción

    Returns:
    predictions -- predicciones binarias (0/1)
    confidence -- nivel de confianza
    \"\"\"

    # 1. Ventanas deslizantes con features TDA
    # (approx. 10 lines)
    # YOUR CODE STARTS HERE










    # YOUR CODE ENDS HERE

    # 2. Detección de anomalías
    # (approx. 5 lines)
    # YOUR CODE STARTS HERE





    # YOUR CODE ENDS HERE

    return predictions, confidence"""))

header5 = nbf.v4.new_markdown_cell("""## Ejercicios Avanzados: Reconstrucción y Predicción

---""")
nb5.cells.insert(insert_idx5, header5)
for i, cell in enumerate(new_cells5):
    nb5.cells.insert(insert_idx5 + 1 + i, cell)

with open('notebooks/05_Series_Temporales_EEG_v2.ipynb', 'w') as f:
    nbf.write(nb5, f)

print("✅ Tutorial 5 expandido: 6 ejercicios totales (agregados 3 nuevos)")

# =========================================================================
# TUTORIAL 6: Caso Estudio - Agregar 2 ejercicios
# =========================================================================
print("\n🔧 Expandiendo Tutorial 6...")

with open('notebooks/06_Caso_Estudio_Epilepsia_v2.ipynb', 'r') as f:
    nb6 = nbf.read(f, as_version=4)

insert_idx6 = len(nb6.cells) - 1
new_cells6 = []

# Ejercicio 4
new_cells6.append(nbf.v4.new_markdown_cell("""### Ejercicio 4 - feature_importance_analysis

Analiza importancia de características para clasificación.

**Objetivo:** Entender qué features TDA son más discriminativas"""))

new_cells6.append(nbf.v4.new_code_cell("""# EJERCICIO 4: Análisis de Importancia

def feature_importance_analysis(clf, feature_names):
    \"\"\"
    Analiza y visualiza importancia de características.

    Arguments:
    clf -- clasificador entrenado (Random Forest)
    feature_names -- nombres de las características

    Returns:
    importance_dict -- dict {feature: importance}
    top_features -- top 5 features más importantes
    \"\"\"

    # 1. Extraer importancias del modelo
    # (approx. 3 lines)
    # YOUR CODE STARTS HERE



    # YOUR CODE ENDS HERE

    # 2. Visualizar
    # (approx. 8 lines)
    # Gráfico de barras ordenado
    # YOUR CODE STARTS HERE








    # YOUR CODE ENDS HERE

    return importance_dict, top_features"""))

# Ejercicio 5
new_cells6.append(nbf.v4.new_markdown_cell("""### Ejercicio 5 - cross_validate_pipeline

Valida el pipeline completo con cross-validation.

**Objetivo:** Asegurar robustez del modelo"""))

new_cells6.append(nbf.v4.new_code_cell("""# EJERCICIO 5: Cross-Validation Completa

def cross_validate_pipeline(X, y, n_folds=5):
    \"\"\"
    Valida pipeline con k-fold cross-validation.

    Arguments:
    X -- características
    y -- etiquetas
    n_folds -- número de folds

    Returns:
    cv_scores -- scores de cada fold
    mean_score -- score promedio
    std_score -- desviación estándar
    \"\"\"
    from sklearn.model_selection import cross_val_score
    from sklearn.ensemble import RandomForestClassifier

    # 1. Setup de cross-validation
    # (approx. 3 lines)
    # YOUR CODE STARTS HERE



    # YOUR CODE ENDS HERE

    # 2. Ejecutar CV y calcular estadísticas
    # (approx. 5 lines)
    # YOUR CODE STARTS HERE





    # YOUR CODE ENDS HERE

    return cv_scores, mean_score, std_score"""))

header6 = nbf.v4.new_markdown_cell("""## Ejercicios Avanzados: Validación y Análisis

---""")
nb6.cells.insert(insert_idx6, header6)
for i, cell in enumerate(new_cells6):
    nb6.cells.insert(insert_idx6 + 1 + i, cell)

with open('notebooks/06_Caso_Estudio_Epilepsia_v2.ipynb', 'w') as f:
    nbf.write(nb6, f)

print("✅ Tutorial 6 expandido: 5 ejercicios totales (agregados 2 nuevos)")

print("\n" + "="*60)
print("🎉 EXPANSIÓN COMPLETADA")
print("="*60)
print("\nResumen:")
print("• Tutorial 1: 7 ejercicios (4 → 7)")
print("• Tutorial 2: 7 ejercicios (4 → 7)")
print("• Tutorial 3: 6 ejercicios (3 → 6)")
print("• Tutorial 4: 5 ejercicios (3 → 5)")
print("• Tutorial 5: 6 ejercicios (3 → 6)")
print("• Tutorial 6: 5 ejercicios (3 → 5)")
print("\n📊 TOTAL: 36 ejercicios (antes: 20)")
print("    Incremento: +16 ejercicios (+80%)")

EOF

chmod +x expand_all_tutorials.sh
bash expand_all_tutorials.sh

