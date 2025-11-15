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
