"""
Tests automáticos para los tutoriales de TDA
Inspirado en el estilo de Coursera Deep Learning Specialization
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform

def test_build_simplicial_complex(target):
    """
    Test para la construcción de complejo simplicial
    """
    print("Ejecutando tests para build_simplicial_complex...")

    # Test 1: Red simple de 4 puntos
    points = np.array([
        [0, 0],
        [1, 0],
        [0.5, 0.8],
        [2, 0]
    ])
    epsilon = 1.0

    edges, triangles = target(points, epsilon)

    # Verificaciones
    assert isinstance(edges, list), "❌ edges debe ser una lista"
    assert isinstance(triangles, list), "❌ triangles debe ser una lista"
    assert len(edges) == 4, f"❌ Esperado 4 aristas con ε=1.0, obtuviste {len(edges)}"
    assert len(triangles) == 1, f"❌ Esperado 1 triángulo con ε=1.0, obtuviste {len(triangles)}"

    # Test 2: Epsilon muy pequeño (sin conexiones)
    edges_small, triangles_small = target(points, epsilon=0.1)
    assert len(edges_small) == 0, f"❌ Con ε=0.1 no debería haber aristas, obtuviste {len(edges_small)}"
    assert len(triangles_small) == 0, f"❌ Con ε=0.1 no debería haber triángulos"

    # Test 3: Epsilon muy grande (todo conectado)
    edges_large, triangles_large = target(points, epsilon=5.0)
    expected_edges = 6  # C(4,2) = 6 aristas posibles
    assert len(edges_large) == expected_edges, f"❌ Con ε=5.0 esperado {expected_edges} aristas"

    print("\033[92m✅ Todos los tests de build_simplicial_complex pasaron!\033[0m")


def test_compute_betti_numbers(target):
    """
    Test para el cálculo de números de Betti
    """
    print("Ejecutando tests para compute_betti_numbers...")

    # Test 1: Círculo (debe tener β₁ = 1)
    from sklearn.datasets import make_circles
    circle_points, _ = make_circles(n_samples=100, noise=0.05, factor=0.5, random_state=42)
    # Solo círculo exterior
    circle_points = circle_points[circle_points[:, 0]**2 + circle_points[:, 1]**2 > 0.1]

    result = target(circle_points, max_epsilon=1.0, num_steps=50)
    epsilons, betti_0, betti_1, betti_2 = result

    # Verificaciones básicas
    assert len(epsilons) == 50, f"❌ Esperado 50 valores de epsilon, obtuviste {len(epsilons)}"
    assert len(betti_0) == 50, "❌ betti_0 debe tener 50 valores"
    assert len(betti_1) == 50, "❌ betti_1 debe tener 50 valores"

    # El círculo debe converger a β₀ = 1
    assert betti_0[-1] == 1, f"❌ Al final β₀ debe ser 1 (una componente), obtuviste {betti_0[-1]}"

    # Debe haber detectado al menos un ciclo en algún punto
    assert np.max(betti_1) >= 1, f"❌ Debería detectar al menos un ciclo (β₁≥1), máximo fue {np.max(betti_1)}"

    print("\033[92m✅ Todos los tests de compute_betti_numbers pasaron!\033[0m")


def test_generate_neural_network(target):
    """
    Test para generación de red neuronal sintética
    """
    print("Ejecutando tests para generate_neural_network...")

    n_neurons = 50
    neurons = target(n_neurons=n_neurons, connectivity=0.3, noise_level=0.1)

    # Verificaciones
    assert neurons.shape[0] == n_neurons + 1, f"❌ Esperado {n_neurons+1} neuronas (incluyendo puente), obtuviste {neurons.shape[0]}"
    assert neurons.shape[1] == 2, f"❌ Debe ser 2D, obtuviste {neurons.shape[1]} dimensiones"

    # Verificar que hay dos comunidades separadas
    # Las primeras n_neurons//2 deben estar cerca de [0,0]
    # Las siguientes n_neurons//2 deben estar cerca de [3,0]
    community1 = neurons[:n_neurons//2, :]
    community2 = neurons[n_neurons//2:n_neurons, :]

    mean1 = np.mean(community1, axis=0)
    mean2 = np.mean(community2, axis=0)

    # Las comunidades deben estar separadas
    distance_between_communities = np.linalg.norm(mean1 - mean2)
    assert distance_between_communities > 1.5, f"❌ Las comunidades deben estar separadas (distancia > 1.5), obtuviste {distance_between_communities:.2f}"

    print("\033[92m✅ Todos los tests de generate_neural_network pasaron!\033[0m")


def test_generate_brain_state(target):
    """
    Test para generación de estados cerebrales
    """
    print("Ejecutando tests para generate_brain_state...")

    n_neurons = 100

    # Test 1: Estado de reposo
    resting = target(state_type='resting', n_neurons=n_neurons)
    assert resting.shape == (n_neurons, 3), f"❌ Forma incorrecta para 'resting': {resting.shape}"

    # Test 2: Estado activo (esfera)
    active = target(state_type='active', n_neurons=n_neurons)
    assert active.shape == (n_neurons, 3), f"❌ Forma incorrecta para 'active': {active.shape}"

    # El estado activo debe formar una esfera (radio ~1)
    radii = np.linalg.norm(active, axis=1)
    mean_radius = np.mean(radii)
    assert 0.7 < mean_radius < 1.3, f"❌ Estado activo debe tener radio ~1, obtuviste {mean_radius:.2f}"

    # Los dos estados deben ser diferentes
    distance = np.linalg.norm(np.mean(resting, axis=0) - np.mean(active, axis=0))
    assert distance > 0.1, "❌ Los estados 'resting' y 'active' deben ser diferentes"

    print("\033[92m✅ Todos los tests de generate_brain_state pasaron!\033[0m")


def test_extract_features_from_diagram(target):
    """
    Test para extracción de características de diagrama de persistencia
    """
    print("Ejecutando tests para extract_features_from_diagram...")

    # Crear diagrama sintético
    diagram = np.array([
        [0.1, 0.5],
        [0.2, 0.4],
        [0.3, 0.8],
        [0.0, np.inf]  # Punto infinito
    ])

    features = target(diagram, dim=1)

    # Verificaciones
    assert 'n_features' in features, "❌ Debe retornar 'n_features'"
    assert 'max_persistence' in features, "❌ Debe retornar 'max_persistence'"
    assert 'mean_persistence' in features, "❌ Debe retornar 'mean_persistence'"

    # El diagrama tiene 3 features finitas
    assert features['n_features'] == 3, f"❌ Esperado 3 features, obtuviste {features['n_features']}"

    # Persistencia máxima = 0.8 - 0.3 = 0.5
    assert np.isclose(features['max_persistence'], 0.5), f"❌ max_persistence debería ser 0.5, obtuviste {features['max_persistence']}"

    print("\033[92m✅ Todos los tests de extract_features_from_diagram pasaron!\033[0m")


# Test para diagrama vacío
def test_empty_diagram_handling(target):
    """
    Test para manejo de diagramas vacíos
    """
    print("Ejecutando tests para manejo de diagramas vacíos...")

    empty_diagram = np.array([]).reshape(0, 2)
    features = target(empty_diagram, dim=1)

    assert features['n_features'] == 0, "❌ Diagrama vacío debe tener 0 features"
    assert features['max_persistence'] == 0, "❌ max_persistence debe ser 0"
    assert features['mean_persistence'] == 0, "❌ mean_persistence debe ser 0"

    print("\033[92m✅ Test de diagrama vacío pasó!\033[0m")


# Función de utilidad para mostrar progreso
def run_all_tests_tutorial1(functions_dict):
    """
    Ejecuta todos los tests para el Tutorial 1

    Arguments:
    functions_dict -- diccionario con las funciones implementadas
                     {'build_simplicial_complex': func1, ...}
    """
    print("\n" + "="*60)
    print("EJECUTANDO SUITE DE TESTS - TUTORIAL 1")
    print("="*60 + "\n")

    tests_passed = 0
    tests_failed = 0

    # Lista de tests a ejecutar
    test_suite = [
        ('build_simplicial_complex', test_build_simplicial_complex),
        ('compute_betti_numbers', test_compute_betti_numbers),
        ('generate_neural_network', test_generate_neural_network),
        ('generate_brain_state', test_generate_brain_state),
    ]

    for func_name, test_func in test_suite:
        if func_name in functions_dict:
            try:
                test_func(functions_dict[func_name])
                tests_passed += 1
            except AssertionError as e:
                print(f"\033[91m{str(e)}\033[0m")
                tests_failed += 1
            except Exception as e:
                print(f"\033[91m❌ Error inesperado en {func_name}: {str(e)}\033[0m")
                tests_failed += 1
        else:
            print(f"\033[93m⚠️  Función '{func_name}' no encontrada en el diccionario\033[0m")

    print("\n" + "="*60)
    print(f"RESUMEN: {tests_passed} tests pasaron, {tests_failed} fallaron")
    print("="*60 + "\n")

    if tests_failed == 0:
        print("\033[92m🎉 ¡FELICITACIONES! Todos los tests pasaron exitosamente\033[0m")

    return tests_passed, tests_failed


# ============================================================================
# TESTS PARA TUTORIAL 2: Homología Persistente Avanzada
# ============================================================================

def test_generate_brain_state_realistic(target):
    """
    Test para generación de estados cerebrales realistas
    """
    print("Ejecutando tests para generate_brain_state_realistic...")

    n_neurons = 100

    # Test cada tipo de estado
    states = ['sleep', 'wakeful', 'attention', 'memory']

    for state_type in states:
        data = target(state_type=state_type, n_neurons=n_neurons, noise=0.1)
        assert data.shape == (n_neurons, 5), f"❌ Forma incorrecta para '{state_type}': {data.shape}"

    # El estado 'memory' debe tener estructura cíclica
    memory_data = target(state_type='memory', n_neurons=n_neurons, noise=0.05)
    # Verificar que hay variación en las columnas (no todo ceros)
    assert np.std(memory_data) > 0.1, "❌ Estado 'memory' debe tener variación significativa"

    # Estados diferentes deben ser distinguibles
    sleep_data = target(state_type='sleep', n_neurons=n_neurons, noise=0.1)
    wakeful_data = target(state_type='wakeful', n_neurons=n_neurons, noise=0.1)

    distance = np.linalg.norm(np.mean(sleep_data, axis=0) - np.mean(wakeful_data, axis=0))
    assert distance > 0.1, "❌ Estados 'sleep' y 'wakeful' deben ser diferentes"

    print("\033[92m✅ Todos los tests de generate_brain_state_realistic pasaron!\033[0m")


def test_generate_spike_trains(target):
    """
    Test para generación de spike trains
    """
    print("Ejecutando tests para generate_spike_trains...")

    n_neurons = 20
    duration = 1000

    # Test cada patrón
    patterns = ['random', 'synchronized', 'sequential']

    for pattern in patterns:
        spike_trains = target(n_neurons=n_neurons, duration=duration,
                            base_rate=5.0, pattern_type=pattern)

        assert spike_trains.shape == (n_neurons, duration), \
            f"❌ Forma incorrecta para '{pattern}': {spike_trains.shape}"

        # Verificar que hay spikes
        total_spikes = np.sum(spike_trains)
        assert total_spikes > 0, f"❌ Patrón '{pattern}' debe generar spikes"

    # Patrón sincronizado debe tener alta correlación entre neuronas
    sync_trains = target(n_neurons=10, duration=500, pattern_type='synchronized')
    correlations = []
    for i in range(5):
        for j in range(i+1, 5):
            corr = np.corrcoef(sync_trains[i], sync_trains[j])[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)

    if len(correlations) > 0:
        mean_corr = np.mean(correlations)
        assert mean_corr > 0.3, f"❌ Patrón sincronizado debe tener correlación > 0.3, obtuviste {mean_corr:.2f}"

    print("\033[92m✅ Todos los tests de generate_spike_trains pasaron!\033[0m")


def test_spike_trains_to_state_space(target):
    """
    Test para conversión de spike trains a espacio de estados
    """
    print("Ejecutando tests para spike_trains_to_state_space...")

    n_neurons = 15
    duration = 1000
    bin_size = 50
    stride = 25

    # Crear spike trains sintéticos
    spike_trains = np.random.poisson(0.05, size=(n_neurons, duration))

    # Convertir
    state_space = target(spike_trains, bin_size=bin_size, stride=stride)

    # Verificar forma
    expected_bins = (duration - bin_size) // stride + 1
    assert state_space.shape[0] == expected_bins, \
        f"❌ Esperado {expected_bins} bins, obtuviste {state_space.shape[0]}"
    assert state_space.shape[1] == n_neurons, \
        f"❌ Esperado {n_neurons} neuronas, obtuviste {state_space.shape[1]}"

    # Valores deben ser no-negativos (son conteos de spikes)
    assert np.all(state_space >= 0), "❌ Los conteos de spikes deben ser no-negativos"

    print("\033[92m✅ Todos los tests de spike_trains_to_state_space pasaron!\033[0m")


def test_extract_topological_features_tutorial2(target):
    """
    Test para extracción de características topológicas (Tutorial 2)
    """
    print("Ejecutando tests para extract_topological_features...")

    # Crear diagrama sintético con dos dimensiones
    diagram_0 = np.array([
        [0.0, 0.5],
        [0.1, 0.3],
        [0.0, np.inf]
    ])

    diagram_1 = np.array([
        [0.2, 0.8],
        [0.3, 0.6],
        [0.1, 0.5]
    ])

    diagram = [diagram_0, diagram_1]

    # Extraer características de H₁
    features = target(diagram, dim=1)

    # Verificaciones
    required_keys = ['n_features', 'max_persistence', 'mean_persistence',
                     'std_persistence', 'total_persistence', 'entropy']

    for key in required_keys:
        assert key in features, f"❌ Debe retornar '{key}'"

    # Número de features (excluyendo infinitos)
    assert features['n_features'] == 3, \
        f"❌ Esperado 3 features en H₁, obtuviste {features['n_features']}"

    # Max persistence = 0.8 - 0.2 = 0.6
    assert np.isclose(features['max_persistence'], 0.6, atol=0.01), \
        f"❌ max_persistence debería ser 0.6, obtuviste {features['max_persistence']}"

    # Entropía debe ser positiva
    assert features['entropy'] > 0, \
        f"❌ Entropía debe ser positiva, obtuviste {features['entropy']}"

    print("\033[92m✅ Todos los tests de extract_topological_features pasaron!\033[0m")


def run_all_tests_tutorial2(functions_dict):
    """
    Ejecuta todos los tests para el Tutorial 2

    Arguments:
    functions_dict -- diccionario con las funciones implementadas
    """
    print("\n" + "="*60)
    print("EJECUTANDO SUITE DE TESTS - TUTORIAL 2")
    print("="*60 + "\n")

    tests_passed = 0
    tests_failed = 0

    # Lista de tests a ejecutar
    test_suite = [
        ('generate_brain_state_realistic', test_generate_brain_state_realistic),
        ('generate_spike_trains', test_generate_spike_trains),
        ('spike_trains_to_state_space', test_spike_trains_to_state_space),
        ('extract_topological_features', test_extract_topological_features_tutorial2),
    ]

    for func_name, test_func in test_suite:
        if func_name in functions_dict:
            try:
                test_func(functions_dict[func_name])
                tests_passed += 1
            except AssertionError as e:
                print(f"\033[91m{str(e)}\033[0m")
                tests_failed += 1
            except Exception as e:
                print(f"\033[91m❌ Error inesperado en {func_name}: {str(e)}\033[0m")
                tests_failed += 1
        else:
            print(f"\033[93m⚠️  Función '{func_name}' no encontrada en el diccionario\033[0m")

    print("\n" + "="*60)
    print(f"RESUMEN: {tests_passed} tests pasaron, {tests_failed} fallaron")
    print("="*60 + "\n")

    if tests_failed == 0:
        print("\033[92m🎉 ¡FELICITACIONES! Todos los tests pasaron exitosamente\033[0m")

    return tests_passed, tests_failed


# ============================================================================
# TESTS PARA TUTORIAL 6: Caso de Estudio Epilepsia
# ============================================================================

def test_preprocess_eeg_tutorial6(target):
    """Test para preprocesamiento de EEG"""
    print("Ejecutando tests para preprocess_eeg...")
    eeg_data = np.random.randn(23, 2560)
    result = target(eeg_data, fs=256)
    assert result.shape == eeg_data.shape, f"❌ Shape incorrecta: {result.shape}"
    assert np.abs(np.mean(result)) < 0.5, "❌ Datos deben estar centrados"
    assert 0.8 < np.std(result) < 1.2, "❌ Datos deben estar normalizados"
    print("\033[92m✅ Todos los tests de preprocess_eeg pasaron!\033[0m")

def test_extract_comprehensive_features_tutorial6(target):
    """Test para extracción de características"""
    print("Ejecutando tests para extract_comprehensive_features...")
    eeg_data = np.random.randn(23, 2560)
    features = target(eeg_data, fs=256)
    assert 'n_cycles' in features or any('cycle' in k for k in features.keys()), "❌ Debe incluir info de ciclos"
    assert len(features) >= 10, f"❌ Esperado >=10 features, obtuviste {len(features)}"
    print("\033[92m✅ Todos los tests de extract_comprehensive_features pasaron!\033[0m")

def test_train_topological_classifier(target):
    """Test para clasificador"""
    print("Ejecutando tests para train_topological_classifier...")
    X = np.random.randn(100, 15)
    y = np.random.randint(0, 2, 100)
    clf, results = target(X, y, test_size=0.3)
    assert clf is not None, "❌ Debe retornar clasificador"
    assert 'accuracy' in results, "❌ results debe incluir 'accuracy'"
    assert 0 <= results['accuracy'] <= 1, f"❌ Accuracy inválida: {results['accuracy']}"
    print("\033[92m✅ Todos los tests de train_topological_classifier pasaron!\033[0m")


# ============================================================================
# TESTS PARA TUTORIAL 3: Conectividad Cerebral
# ============================================================================

def test_build_connectivity_matrix(target):
    """Test para construcción de matriz de conectividad"""
    print("Ejecutando tests para build_connectivity_matrix...")
    
    # Test 1: Shape correcto
    timeseries = np.random.randn(30, 200)
    conn_matrix, diagrams = target(timeseries, threshold=0.3)
    assert conn_matrix.shape == (30, 30), f"❌ Shape esperada (30,30), obtuviste {conn_matrix.shape}"
    
    # Test 2: Matriz simétrica
    assert np.allclose(conn_matrix, conn_matrix.T), "❌ Matriz debe ser simétrica"
    
    # Test 3: Valores en rango correcto
    assert np.all(conn_matrix >= -1) and np.all(conn_matrix <= 1), "❌ Valores deben estar en [-1, 1]"
    
    # Test 4: Diagramas de persistencia
    assert len(diagrams) >= 2, "❌ Debe retornar al menos H0 y H1"
    
    print("\033[92m✅ Todos los tests de build_connectivity_matrix pasaron!\033[0m")


def test_detect_communities_topological(target):
    """Test para detección de comunidades"""
    print("Ejecutando tests para detect_communities_topological...")
    
    # Test 1: Clustering básico
    conn_matrix = np.random.rand(60, 60)
    conn_matrix = (conn_matrix + conn_matrix.T) / 2  # Simétrica
    detected_labels, ari_score = target(conn_matrix, n_clusters=3, true_labels=None)
    
    assert len(detected_labels) == 60, f"❌ Debe retornar 60 etiquetas, obtuviste {len(detected_labels)}"
    assert len(set(detected_labels)) <= 3, "❌ Debe haber máximo 3 clusters"
    
    # Test 2: Con etiquetas verdaderas
    true_labels = np.array([0]*20 + [1]*20 + [2]*20)
    detected_labels, ari_score = target(conn_matrix, n_clusters=3, true_labels=true_labels)
    assert ari_score is not None, "❌ Debe calcular ARI score"
    assert -1 <= ari_score <= 1, f"❌ ARI debe estar en [-1,1], obtuviste {ari_score}"
    
    print("\033[92m✅ Todos los tests de detect_communities_topological pasaron!\033[0m")


def test_compare_states_topologically(target):
    """Test para comparación de estados"""
    print("Ejecutando tests para compare_states_topologically...")
    
    # Crear estados sintéticos
    states = {
        'state1': np.random.randn(50, 200),
        'state2': np.random.randn(50, 200),
        'state3': np.random.randn(50, 200)
    }
    
    distance_matrix, all_diagrams = target(states)
    
    # Test 1: Shape de matriz de distancias
    assert distance_matrix.shape == (3, 3), f"❌ Shape esperada (3,3), obtuviste {distance_matrix.shape}"
    
    # Test 2: Matriz simétrica
    assert np.allclose(distance_matrix, distance_matrix.T), "❌ Matriz debe ser simétrica"
    
    # Test 3: Diagonal cero
    assert np.allclose(np.diag(distance_matrix), 0), "❌ Diagonal debe ser cero"
    
    # Test 4: Diagramas para cada estado
    assert len(all_diagrams) == 3, f"❌ Debe haber 3 diagramas, obtuviste {len(all_diagrams)}"
    
    print("\033[92m✅ Todos los tests de compare_states_topologically pasaron!\033[0m")


# ============================================================================
# TESTS PARA TUTORIAL 4: Mapper Algorithm
# ============================================================================

def test_compute_filter_function(target):
    """Test para funciones de filtro"""
    print("Ejecutando tests para compute_filter_function...")
    
    data = np.random.randn(100, 10)
    
    # Test 1: Filtro PCA
    filter_pca = target(data, method='pca')
    assert filter_pca.shape == (100,), f"❌ Shape esperada (100,), obtuviste {filter_pca.shape}"
    
    # Test 2: Filtro Density
    filter_density = target(data, method='density', n_neighbors=5)
    assert filter_density.shape == (100,), "❌ Shape incorrecta para density"
    assert np.all(filter_density >= 0), "❌ Densidades deben ser positivas"
    
    # Test 3: Filtro Coordinate
    filter_coord = target(data, method='coordinate', coord_idx=0)
    assert np.allclose(filter_coord, data[:, 0]), "❌ Coordinate filter debe retornar columna seleccionada"
    
    print("\033[92m✅ Todos los tests de compute_filter_function pasaron!\033[0m")


def test_build_mapper_graph(target):
    """Test para construcción de grafo Mapper"""
    print("Ejecutando tests para build_mapper_graph...")
    
    data = np.random.randn(200, 5)
    filter_values = np.random.randn(200)
    
    G, nodes_data = target(data, filter_values, n_intervals=10, overlap=0.3)
    
    # Test 1: Tipo de grafo
    assert isinstance(G, nx.Graph), "❌ Debe retornar grafo de NetworkX"
    
    # Test 2: Nodos existen
    assert G.number_of_nodes() > 0, "❌ Grafo debe tener al menos un nodo"
    
    # Test 3: Diccionario de nodos
    assert isinstance(nodes_data, dict), "❌ nodes_data debe ser diccionario"
    
    # Test 4: Consistencia
    assert len(nodes_data) == G.number_of_nodes(), "❌ Número de nodos inconsistente"
    
    print("\033[92m✅ Todos los tests de build_mapper_graph pasaron!\033[0m")


def test_visualize_mapper(target):
    """Test para visualización de Mapper"""
    print("Ejecutando tests para visualize_mapper...")
    
    # Crear grafo simple
    G = nx.Graph()
    G.add_node(0, size=10)
    G.add_node(1, size=15)
    G.add_edge(0, 1)
    
    nodes_data = {0: np.array([0, 1, 2]), 1: np.array([3, 4, 5])}
    filter_values = np.array([0.5, 0.6, 0.7, 1.0, 1.1, 1.2])
    
    fig, ax = target(G, nodes_data, filter_values, title="Test")
    
    # Test 1: Retorna figura y eje
    assert fig is not None, "❌ Debe retornar figura"
    assert ax is not None, "❌ Debe retornar eje"
    
    # Test 2: Título correcto
    assert ax.get_title() == "Test", "❌ Título incorrecto"
    
    plt.close(fig)
    
    print("\033[92m✅ Todos los tests de visualize_mapper pasaron!\033[0m")


# ============================================================================
# TESTS PARA TUTORIAL 5: Series Temporales EEG
# ============================================================================

def test_takens_embedding(target):
    """Test para embedding de Takens"""
    print("Ejecutando tests para takens_embedding...")
    
    # Señal sintética
    signal = np.sin(np.linspace(0, 4*np.pi, 1000))
    
    # Test 1: Con delay especificado
    embedded, delay_used = target(signal, delay=10, dimension=3)
    assert embedded.shape[1] == 3, f"❌ Dimensión esperada 3, obtuviste {embedded.shape[1]}"
    assert delay_used == 10, f"❌ Delay usado debe ser 10, obtuviste {delay_used}"
    
    # Test 2: Delay automático
    embedded_auto, delay_auto = target(signal, delay=None, dimension=3)
    assert embedded_auto.shape[1] == 3, "❌ Dimensión incorrecta"
    assert delay_auto > 0, f"❌ Delay automático debe ser > 0, obtuviste {delay_auto}"
    
    # Test 3: Shape correcto
    expected_length = len(signal) - (3 - 1) * delay_used
    assert embedded.shape[0] == expected_length, f"❌ Shape incorrecta"
    
    print("\033[92m✅ Todos los tests de takens_embedding pasaron!\033[0m")


def test_sliding_window_persistence(target):
    """Test para análisis con ventanas deslizantes"""
    print("Ejecutando tests para sliding_window_persistence...")
    
    # Señal sintética
    signal = np.random.randn(2000)
    
    time_pts, n_cycles, max_pers = target(signal, window_size=500, stride=100, fs=250)
    
    # Test 1: Longitudes consistentes
    assert len(time_pts) == len(n_cycles) == len(max_pers), "❌ Arrays deben tener misma longitud"
    
    # Test 2: Valores razonables
    assert np.all(n_cycles >= 0), "❌ Número de ciclos debe ser >= 0"
    assert np.all(max_pers >= 0), "❌ Persistencia debe ser >= 0"
    
    # Test 3: Número esperado de ventanas
    expected_windows = (len(signal) - 500) // 100 + 1
    assert len(time_pts) <= expected_windows, f"❌ Demasiadas ventanas"
    
    print("\033[92m✅ Todos los tests de sliding_window_persistence pasaron!\033[0m")


def test_classify_states_with_tda(target):
    """Test para clasificación con TDA"""
    print("Ejecutando tests para classify_states_with_tda...")
    
    # Dataset sintético
    signals_dict = {
        'normal': [np.random.randn(1250) for _ in range(20)],
        'seizure': [np.random.randn(1250) * 2 for _ in range(20)],
        'sleep': [np.random.randn(1250) * 0.5 for _ in range(20)]
    }
    
    clf, accuracy, report = target(signals_dict, test_size=0.3)
    
    # Test 1: Clasificador entrenado
    assert clf is not None, "❌ Debe retornar clasificador"
    
    # Test 2: Accuracy en rango válido
    assert 0 <= accuracy <= 1, f"❌ Accuracy inválida: {accuracy}"
    
    # Test 3: Reporte existe
    assert report is not None, "❌ Debe retornar reporte"
    
    print("\033[92m✅ Todos los tests de classify_states_with_tda pasaron!\033[0m")


# ============================================================================
# FUNCIÓN HELPER PARA EJECUTAR TODOS LOS TESTS DE TUTORIAL 3
# ============================================================================

def run_all_tests_tutorial3(functions_dict):
    """Ejecuta todos los tests para Tutorial 3"""
    print("\n" + "="*60)
    print("EJECUTANDO TESTS PARA TUTORIAL 3: Conectividad Cerebral")
    print("="*60 + "\n")
    
    test_functions = [
        ('build_connectivity_matrix', test_build_connectivity_matrix),
        ('detect_communities_topological', test_detect_communities_topological),
        ('compare_states_topologically', test_compare_states_topologically)
    ]
    
    tests_passed = 0
    tests_failed = 0
    
    for func_name, test_func in test_functions:
        if func_name in functions_dict:
            try:
                test_func(functions_dict[func_name])
                tests_passed += 1
            except AssertionError as e:
                print(f"\033[91m{e}\033[0m")
                tests_failed += 1
        else:
            print(f"\033[93m⚠️  Función '{func_name}' no encontrada\033[0m")
    
    print(f"\nRESUMEN: {tests_passed} pasaron, {tests_failed} fallaron\n")
    return tests_passed, tests_failed


def run_all_tests_tutorial4(functions_dict):
    """Ejecuta todos los tests para Tutorial 4"""
    print("\n" + "="*60)
    print("EJECUTANDO TESTS PARA TUTORIAL 4: Mapper Algorithm")
    print("="*60 + "\n")
    
    test_functions = [
        ('compute_filter_function', test_compute_filter_function),
        ('build_mapper_graph', test_build_mapper_graph),
        ('visualize_mapper', test_visualize_mapper)
    ]
    
    tests_passed = 0
    tests_failed = 0
    
    for func_name, test_func in test_functions:
        if func_name in functions_dict:
            try:
                test_func(functions_dict[func_name])
                tests_passed += 1
            except AssertionError as e:
                print(f"\033[91m{e}\033[0m")
                tests_failed += 1
        else:
            print(f"\033[93m⚠️  Función '{func_name}' no encontrada\033[0m")
    
    print(f"\nRESUMEN: {tests_passed} pasaron, {tests_failed} fallaron\n")
    return tests_passed, tests_failed


def run_all_tests_tutorial5(functions_dict):
    """Ejecuta todos los tests para Tutorial 5"""
    print("\n" + "="*60)
    print("EJECUTANDO TESTS PARA TUTORIAL 5: Series Temporales EEG")
    print("="*60 + "\n")
    
    test_functions = [
        ('takens_embedding', test_takens_embedding),
        ('sliding_window_persistence', test_sliding_window_persistence),
        ('classify_states_with_tda', test_classify_states_with_tda)
    ]
    
    tests_passed = 0
    tests_failed = 0
    
    for func_name, test_func in test_functions:
        if func_name in functions_dict:
            try:
                test_func(functions_dict[func_name])
                tests_passed += 1
            except AssertionError as e:
                print(f"\033[91m{e}\033[0m")
                tests_failed += 1
        else:
            print(f"\033[93m⚠️  Función '{func_name}' no encontrada\033[0m")
    
    print(f"\nRESUMEN: {tests_passed} pasaron, {tests_failed} fallaron\n")
    return tests_passed, tests_failed


