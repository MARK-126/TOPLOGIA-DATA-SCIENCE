# ============================================================================
# TESTS PARA NUEVOS EJERCICIOS - TUTORIAL 2
# ============================================================================

def test_compute_wasserstein_distance(target):
    """Test para compute_wasserstein_distance"""
    print("Ejecutando tests para compute_wasserstein_distance...")

    # Test 1: Diagramas idénticos
    dgm1 = np.array([[0.5, 1.0], [1.0, 2.0]])
    dgm2 = np.array([[0.5, 1.0], [1.0, 2.0]])
    dist = target(dgm1, dgm2)
    assert dist < 0.01, f"❌ Distancia entre diagramas idénticos debe ser ~0, obtuviste {dist}"

    # Test 2: Diagramas diferentes
    dgm3 = np.array([[0.5, 3.0], [1.5, 4.0]])
    dist2 = target(dgm1, dgm3)
    assert dist2 > 0.5, "❌ Distancia entre diagramas diferentes debe ser significativa"
    assert isinstance(dist2, (int, float)), "❌ Debe retornar un número"

    print("\033[92m✅ Todos los tests de compute_wasserstein_distance pasaron!\033[0m")


def test_detect_temporal_changes(target):
    """Test para detect_temporal_changes"""
    print("Ejecutando tests para detect_temporal_changes...")

    # Test 1: Señal con cambio abrupto
    t = np.linspace(0, 10, 500)
    signal = np.concatenate([np.sin(t[:250]), 2*np.sin(3*t[250:])])

    change_points, distances = target(signal, window_size=50, threshold=0.5)

    assert isinstance(change_points, list), "❌ change_points debe ser lista"
    assert isinstance(distances, np.ndarray), "❌ distances debe ser array"
    assert len(change_points) >= 1, "❌ Debería detectar al menos 1 cambio"
    assert all(0 <= cp < len(signal) for cp in change_points), "❌ Índices fuera de rango"

    print("\033[92m✅ Todos los tests de detect_temporal_changes pasaron!\033[0m")


def test_classify_spike_patterns(target):
    """Test para classify_spike_patterns"""
    print("Ejecutando tests para classify_spike_patterns...")

    # Test 1: Dataset sintético
    np.random.seed(42)
    spike_trains_list = []
    labels = []

    # Clase 0: Baja frecuencia
    for _ in range(20):
        st = np.random.poisson(5, 100)
        spike_trains_list.append(st)
        labels.append(0)

    # Clase 1: Alta frecuencia
    for _ in range(20):
        st = np.random.poisson(15, 100)
        spike_trains_list.append(st)
        labels.append(1)

    labels = np.array(labels)
    classifier, accuracy, predictions = target(spike_trains_list, labels, test_size=0.3)

    assert 0.0 <= accuracy <= 1.0, f"❌ Accuracy debe estar en [0,1], obtuviste {accuracy}"
    assert accuracy > 0.5, "❌ Accuracy debe ser mejor que azar"
    assert len(predictions) == int(0.3 * len(labels)), "❌ Número de predicciones incorrecto"

    print("\033[92m✅ Todos los tests de classify_spike_patterns pasaron!\033[0m")


# ============================================================================
# TESTS PARA NUEVOS EJERCICIOS - TUTORIAL 3
# ============================================================================

def test_compute_graph_features(target):
    """Test para compute_graph_features"""
    print("Ejecutando tests para compute_graph_features...")

    # Test 1: Matriz de conectividad sintética
    np.random.seed(42)
    n_nodes = 20
    connectivity_matrix = np.random.rand(n_nodes, n_nodes)
    connectivity_matrix = (connectivity_matrix + connectivity_matrix.T) / 2  # Simétrica
    np.fill_diagonal(connectivity_matrix, 0)

    features = target(connectivity_matrix, threshold=0.3)

    assert isinstance(features, dict), "❌ Debe retornar diccionario"
    required_keys = ['clustering_coeff', 'betti_0', 'betti_1', 'persistence_entropy']
    for key in required_keys:
        assert key in features, f"❌ Falta la clave '{key}' en features"

    assert features['clustering_coeff'] >= 0, "❌ Clustering coeff debe ser >= 0"
    assert features['betti_0'] >= 1, "❌ Betti 0 debe ser >= 1"

    print("\033[92m✅ Todos los tests de compute_graph_features pasaron!\033[0m")


def test_find_critical_nodes(target):
    """Test para find_critical_nodes"""
    print("Ejecutando tests para find_critical_nodes...")

    # Test 1: Red con hub central
    n_nodes = 15
    connectivity_matrix = np.zeros((n_nodes, n_nodes))
    # Nodo 0 conectado a todos (hub)
    connectivity_matrix[0, 1:] = 0.8
    connectivity_matrix[1:, 0] = 0.8
    # Conexiones aleatorias entre otros nodos
    for i in range(1, n_nodes):
        for j in range(i+1, n_nodes):
            if np.random.rand() > 0.7:
                connectivity_matrix[i, j] = np.random.rand()
                connectivity_matrix[j, i] = connectivity_matrix[i, j]

    critical_nodes, criticality_scores = target(connectivity_matrix, top_k=5)

    assert isinstance(critical_nodes, np.ndarray), "❌ critical_nodes debe ser array"
    assert len(critical_nodes) == 5, "❌ Debe retornar top_k nodos"
    assert 0 in critical_nodes[:3], "❌ El hub (nodo 0) debería estar en top 3"
    assert len(criticality_scores) == len(connectivity_matrix), "❌ Tamaño incorrecto"

    print("\033[92m✅ Todos los tests de find_critical_nodes pasaron!\033[0m")


def test_track_connectivity_evolution(target):
    """Test para track_connectivity_evolution"""
    print("Ejecutando tests para track_connectivity_evolution...")

    # Test 1: Serie temporal sintética
    np.random.seed(42)
    n_timepoints = 5
    n_channels = 10
    n_samples = 100
    time_series = np.random.randn(n_timepoints, n_channels, n_samples)
    window_indices = list(range(n_timepoints))

    evolution_metrics = target(time_series, window_indices)

    assert isinstance(evolution_metrics, dict), "❌ Debe retornar diccionario"
    required_keys = ['betti_evolution', 'persistence_evolution']
    for key in required_keys:
        assert key in evolution_metrics, f"❌ Falta '{key}'"

    assert len(evolution_metrics['betti_evolution']) == n_timepoints, "❌ Longitud incorrecta"

    print("\033[92m✅ Todos los tests de track_connectivity_evolution pasaron!\033[0m")


# ============================================================================
# TESTS PARA NUEVOS EJERCICIOS - TUTORIAL 4
# ============================================================================

def test_optimize_mapper_parameters(target):
    """Test para optimize_mapper_parameters"""
    print("Ejecutando tests para optimize_mapper_parameters...")

    # Test 1: Dataset sintético
    np.random.seed(42)
    from sklearn.datasets import make_circles
    data, _ = make_circles(n_samples=200, noise=0.05, factor=0.5)
    filter_function = data[:, 0]  # Primera coordenada como filtro

    best_params, best_score, mapper_graph = target(data, filter_function, quality_metric='modularity')

    assert isinstance(best_params, dict), "❌ best_params debe ser diccionario"
    assert 'n_intervals' in best_params, "❌ Falta 'n_intervals'"
    assert 'overlap' in best_params, "❌ Falta 'overlap'"
    assert isinstance(best_score, (int, float)), "❌ best_score debe ser número"
    assert best_score > 0, "❌ Score debe ser positivo"

    print("\033[92m✅ Todos los tests de optimize_mapper_parameters pasaron!\033[0m")


def test_detect_loops_in_mapper(target):
    """Test para detect_loops_in_mapper"""
    print("Ejecutando tests para detect_loops_in_mapper...")

    # Test 1: Crear grafo simple con ciclos
    import networkx as nx
    G = nx.cycle_graph(5)  # Ciclo de 5 nodos
    for node in G.nodes():
        G.nodes[node]['density'] = np.random.rand()
        G.nodes[node]['filter_value'] = np.random.rand()

    cycles, cycle_features = target(G, min_cycle_length=3)

    assert isinstance(cycles, list), "❌ cycles debe ser lista"
    assert isinstance(cycle_features, list), "❌ cycle_features debe ser lista"
    assert len(cycles) == len(cycle_features), "❌ Longitudes inconsistentes"
    assert len(cycles) >= 1, "❌ Debería detectar al menos 1 ciclo"

    if len(cycle_features) > 0:
        assert 'length' in cycle_features[0], "❌ Falta 'length'"
        assert 'persistence' in cycle_features[0], "❌ Falta 'persistence'"

    print("\033[92m✅ Todos los tests de detect_loops_in_mapper pasaron!\033[0m")


# ============================================================================
# TESTS PARA NUEVOS EJERCICIOS - TUTORIAL 5
# ============================================================================

def test_compute_delay_embedding_dim(target):
    """Test para compute_delay_embedding_dim"""
    print("Ejecutando tests para compute_delay_embedding_dim...")

    # Test 1: Señal caótica (Lorenz)
    from scipy.integrate import odeint

    def lorenz(state, t, sigma=10, rho=28, beta=8/3):
        x, y, z = state
        return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

    t = np.linspace(0, 50, 5000)
    state0 = [1.0, 1.0, 1.0]
    states = odeint(lorenz, state0, t)
    signal = states[:, 0]

    optimal_dim, fnn_percentages = target(signal, delay=10, max_dim=8)

    assert isinstance(optimal_dim, int), "❌ optimal_dim debe ser entero"
    assert 1 <= optimal_dim <= 8, f"❌ Dimensión debe estar en [1,8], obtuviste {optimal_dim}"
    assert isinstance(fnn_percentages, np.ndarray), "❌ fnn_percentages debe ser array"
    assert len(fnn_percentages) == 8, "❌ Longitud incorrecta"

    print("\033[92m✅ Todos los tests de compute_delay_embedding_dim pasaron!\033[0m")


def test_reconstruct_attractor(target):
    """Test para reconstruct_attractor"""
    print("Ejecutando tests para reconstruct_attractor...")

    # Test 1: Señal periódica simple
    t = np.linspace(0, 20, 1000)
    signal = np.sin(t) + 0.5 * np.sin(3*t)

    attractor, characteristics = target(signal, delay=5, embedding_dim=3)

    assert isinstance(attractor, np.ndarray), "❌ attractor debe ser array"
    assert attractor.shape[1] == 3, "❌ Dimensión de embedding incorrecta"
    assert isinstance(characteristics, dict), "❌ characteristics debe ser dict"

    required_keys = ['betti_numbers', 'persistence_entropy']
    for key in required_keys:
        assert key in characteristics, f"❌ Falta '{key}'"

    print("\033[92m✅ Todos los tests de reconstruct_attractor pasaron!\033[0m")


def test_predict_next_event(target):
    """Test para predict_next_event"""
    print("Ejecutando tests para predict_next_event...")

    # Test 1: Señal con eventos sintéticos
    np.random.seed(42)
    n_samples = 2000
    signal = np.random.randn(n_samples)

    # Insertar eventos
    event_labels = np.zeros(n_samples)
    event_positions = [500, 1000, 1500]
    for pos in event_positions:
        signal[pos:pos+50] += 5  # Spike en evento
        event_labels[pos:pos+50] = 1

    predictions, probabilities, roc_auc = target(signal, event_labels, window_size=100, prediction_horizon=50)

    assert isinstance(predictions, np.ndarray), "❌ predictions debe ser array"
    assert isinstance(probabilities, np.ndarray), "❌ probabilities debe ser array"
    assert 0.0 <= roc_auc <= 1.0, f"❌ ROC-AUC debe estar en [0,1], obtuviste {roc_auc}"
    assert roc_auc > 0.6, "❌ ROC-AUC debe ser razonable (>0.6)"

    print("\033[92m✅ Todos los tests de predict_next_event pasaron!\033[0m")


# ============================================================================
# TESTS PARA NUEVOS EJERCICIOS - TUTORIAL 6
# ============================================================================

def test_feature_importance_analysis(target):
    """Test para feature_importance_analysis"""
    print("Ejecutando tests para feature_importance_analysis...")

    # Test 1: Dataset sintético
    np.random.seed(42)
    n_samples = 100
    n_features = 15
    X = np.random.randn(n_samples, n_features)
    # Feature 0 es muy discriminativa
    y = (X[:, 0] > 0).astype(int)
    feature_names = [f'feature_{i}' for i in range(n_features)]

    importance_scores, top_features = target(X, y, feature_names)

    assert isinstance(importance_scores, dict), "❌ importance_scores debe ser dict"
    assert isinstance(top_features, list), "❌ top_features debe ser lista"
    assert len(top_features) <= 10, "❌ top_features debe tener máximo 10"
    assert 'feature_0' in top_features[:5], "❌ Feature más importante debería estar en top 5"

    print("\033[92m✅ Todos los tests de feature_importance_analysis pasaron!\033[0m")


def test_cross_validate_pipeline(target):
    """Test para cross_validate_pipeline"""
    print("Ejecutando tests para cross_validate_pipeline...")

    # Test 1: Dataset pequeño sintético
    np.random.seed(42)
    n_epochs = 50
    n_channels = 5
    n_samples = 100

    eeg_data = np.random.randn(n_epochs, n_channels, n_samples)
    # Crear labels balanceadas
    labels = np.array([0] * 25 + [1] * 25)

    cv_results, trained_model = target(eeg_data, labels, cv_folds=3)

    assert isinstance(cv_results, dict), "❌ cv_results debe ser dict"
    required_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    for metric in required_metrics:
        assert metric in cv_results, f"❌ Falta métrica '{metric}'"

    # Verificar formato de métricas (promedio ± std)
    assert isinstance(cv_results['accuracy'], (tuple, list, str)), "❌ Formato de métrica incorrecto"

    print("\033[92m✅ Todos los tests de cross_validate_pipeline pasaron!\033[0m")
