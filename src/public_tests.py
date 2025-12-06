"""
Tests públicos para evaluación interactiva
"""

import numpy as np
from src.tda_utils import print_test_result, EMOJI
from termcolor import colored

def test_build_simplicial_complex(target):
    """
    Test para la construcción de complejo simplicial
    """
    print(f"\n{EMOJI['rocket']} Ejecutando tests para build_simplicial_complex...")

    try:
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
        assert isinstance(edges, list), "edges debe ser una lista"
        assert isinstance(triangles, list), "triangles debe ser una lista"
        
        if len(edges) != 4:
            raise AssertionError(f"Esperado 4 aristas con ε=1.0, obtuviste {len(edges)}")
        if len(triangles) != 1:
            raise AssertionError(f"Esperado 1 triángulo con ε=1.0, obtuviste {len(triangles)}")

        print_test_result("Test 1 (Simple)", True)

        # Test 2: Epsilon muy pequeño (sin conexiones)
        edges_small, triangles_small = target(points, epsilon=0.1)
        if len(edges_small) != 0:
             raise AssertionError(f"Con ε=0.1 no debería haber aristas, obtuviste {len(edges_small)}")
        
        print_test_result("Test 2 (Epsilon pequeño)", True)

        # Test 3: Epsilon muy grande (todo conectado)
        edges_large, triangles_large = target(points, epsilon=5.0)
        expected_edges = 6  # C(4,2) = 6 aristas posibles
        if len(edges_large) != expected_edges:
             raise AssertionError(f"Con ε=5.0 esperado {expected_edges} aristas")

        print_test_result("Test 3 (Epsilon grande)", True)
        print(f"\n{EMOJI['check']} ¡Todos los tests pasaron!")

    except AssertionError as e:
        print_test_result("Test Lógica", False, str(e))
    except Exception as e:
        print_test_result("Error de Ejecución", False, str(e))


def test_compute_betti_numbers(target):
    """
    Test para el cálculo de números de Betti
    """
    print(f"\n{EMOJI['rocket']} Ejecutando tests para compute_betti_numbers...")

    try:
        # Test 1: Círculo (debe tener β₁ = 1)
        from sklearn.datasets import make_circles
        circle_points, _ = make_circles(n_samples=100, noise=0.05, factor=0.5, random_state=42)
        # Solo círculo exterior
        circle_points = circle_points[circle_points[:, 0]**2 + circle_points[:, 1]**2 > 0.1]

        result = target(circle_points, max_epsilon=1.0, num_steps=50)
        
        if len(result) != 4:
             raise AssertionError("La función debe retornar 4 valores: epsilons, betti_0, betti_1, betti_2")
            
        epsilons, betti_0, betti_1, betti_2 = result

        # Verificaciones básicas
        if len(epsilons) != 50:
             raise AssertionError(f"Esperado 50 valores de epsilon, obtuviste {len(epsilons)}")

        # El círculo debe converger a β₀ = 1
        if betti_0[-1] != 1:
             raise AssertionError(f"Al final β₀ debe ser 1 (una componente), obtuviste {betti_0[-1]}")

        # Debe haber detectado al menos un ciclo en algún punto
        if np.max(betti_1) < 1:
             raise AssertionError(f"Debería detectar al menos un ciclo (β₁≥1), máximo fue {np.max(betti_1)}")

        print_test_result("Test Círculo", True)
        print(f"\n{EMOJI['check']} ¡Todos los tests pasaron!")
        
    except AssertionError as e:
        print_test_result("Test Lógica", False, str(e))
    except Exception as e:
        print_test_result("Error de Ejecución", False, str(e))

def test_generate_neural_network(target):
    print(f"\n{EMOJI['rocket']} Ejecutando tests para generate_neural_network...")
    try:
        n_neurons = 50
        neurons = target(n_neurons=n_neurons, connectivity=0.3, noise_level=0.1)

        if neurons.shape[0] != n_neurons + 1:
            raise AssertionError(f"Esperado {n_neurons+1} neuronas, obtuviste {neurons.shape[0]}")
            
        community1 = neurons[:n_neurons//2, :]
        community2 = neurons[n_neurons//2:n_neurons, :]
        dist = np.linalg.norm(np.mean(community1, axis=0) - np.mean(community2, axis=0))
        
        if dist <= 1.5:
            raise AssertionError("Las comunidades no están suficientemente separadas")

        print_test_result("Test Generación", True)
        print(f"\n{EMOJI['check']} ¡Todos los tests pasaron!")
    except Exception as e:
        print_test_result("Error", False, str(e))

def test_extract_topological_features(target):
    """
    Test para extracción de características
    """
    print(f"\n{EMOJI['rocket']} Ejecutando tests para extract_topological_features...")
    
    try:
        # Mock diagram: 2 points in dim 1
        diagram = [
            np.array([]), # Dim 0
            np.array([[0.1, 0.5], [0.2, 0.4]]) # Dim 1
        ]
        
        features = target(diagram, dim=1)
        
        if features['n_features'] != 2:
            raise AssertionError(f"Esperado 2 features, obtuviste {features['n_features']}")
            
        # Max persistence: 0.5-0.1 = 0.4
        if not np.isclose(features['max_persistence'], 0.4):
            raise AssertionError(f"Esperado max_persistence=0.4, obtuviste {features['max_persistence']}")
            
        # Entropy should be > 0
        if features['entropy'] <= 0:
            raise AssertionError("La entropía debería ser positiva para diagrama no vacío")
            
        print_test_result("Test Características Básicas", True)
        print(f"\n{EMOJI['check']} ¡Todos los tests pasaron!")
        
    except AssertionError as e:
        print_test_result("Test Lógica", False, str(e))
    except Exception as e:
        print_test_result("Error de Ejecución", False, str(e))

def test_spike_train_analysis(target_func_gen, target_func_space):
    """
    Test conjunto para pipelines de spike trains
    """
    print(f"\n{EMOJI['rocket']} Ejecutando tests para pipeline de spike trains...")
    
    try:
        # 1. Generate
        spikes = target_func_gen(n_neurons=10, duration=100, pattern_type='synchronized')
        if spikes.shape != (10, 100):
            raise AssertionError(f"Forma incorrecta de spike trains: {spikes.shape}")
            
        # 2. State Space
        space = target_func_space(spikes, bin_size=20, stride=20)
        # bins = 100/20 = 5
        if space.shape != (5, 10):
             raise AssertionError(f"Forma incorrecta de espacio de estados. Esperado (5, 10), obtenido {space.shape}")
             
        print_test_result("Test Pipeline Completo", True)
        print(f"\n{EMOJI['check']} ¡Todos los tests pasaron!")
        
    except Exception as e:
        print_test_result("Error de Ejecución", False, str(e))

def test_connectivity_analysis(target_func_fc, target_func_feats):
    """
    Test para análisis de conectividad
    """
    print(f"\n{EMOJI['rocket']} Ejecutando tests para análisis de conectividad...")
    
    try:
        # Mock timeseries: 2 completely correlated regions
        ts = np.array([
            [1, 2, 3, 4, 5],
            [2, 4, 6, 8, 10], # Perfect corr with 0
            [1, 0, 1, 0, 1]   # Uncorrelated
        ])
        
        # 1. Test Matrix Calculation
        matrix = target_func_fc(ts)
        
        if matrix.shape != (3, 3):
            raise AssertionError(f"Forma incorrecta de matriz de conectividad: {matrix.shape}")
            
        if not np.isclose(matrix[0, 1], 1.0):
            raise AssertionError(f"Correlación perfecta debería ser 1.0, obtuviste {matrix[0,1]}")
            
        # 2. Test Feature Extraction
        feats = target_func_feats(matrix)
        
        required_keys = ['avg_clustering', 'n_cycles']
        for k in required_keys:
            if k not in feats:
                raise AssertionError(f"Falta la característica '{k}'")
                
        print_test_result("Test Conectividad Básica", True)
        print(f"\n{EMOJI['check']} ¡Todos los tests pasaron!")
        
    except Exception as e:
        print_test_result("Error de Ejecución", False, str(e))

def test_community_detection(target_func):
    """
    Test para detección de comunidades
    """
    print(f"\n{EMOJI['rocket']} Ejecutando tests para communities...")
    
    try:
        # Block diagonal matrix (2 communities)
        adj = np.array([
            [1, 0.9, 0, 0],
            [0.9, 1, 0, 0],
            [0, 0, 1, 0.9],
            [0, 0, 0.9, 1]
        ])
        
        labels = target_func(adj, n_clusters=2)
        
        if len(labels) != 4:
            raise AssertionError("Debe retornar 4 etiquetas")
            
        # Node 0 and 1 should be same community
        if labels[0] != labels[1]:
            raise AssertionError("Nodos 0 y 1 deberían estar en la misma comunidad")
            
        # Node 0 and 2 should be different
        if labels[0] == labels[2]:
            raise AssertionError("Nodos 0 y 2 deberían estar en comunidades distintas")
            
        print_test_result("Test Detección Comunidades", True)
        print(f"\n{EMOJI['check']} ¡Todos los tests pasaron!")
        
    except Exception as e:
        print_test_result("Error de Ejecución", False, str(e))

def test_mapper_implementation(target_func):
    """
    Test para implementación de Mapper
    """
    print(f"\n{EMOJI['rocket']} Ejecutando tests para Mapper básico...")
    
    try:
        # Mock simple data: 2 clusters separated
        data = np.concatenate([
            np.random.normal(0, 0.1, (10, 2)),
            np.random.normal(5, 0.1, (10, 2))
        ])
        
        # Simple filter: x-coordinate
        def mock_filter(d): return d[:, 0]
        
        # Call target mapper
        graph, nodes, vals = target_func(
            data, 
            mock_filter,
            n_intervals=2,  # Should easily separate the two blobs
            overlap=0.1
        )
        
        # Checks
        if graph.number_of_nodes() < 2:
            raise AssertionError(f"Debería haber al menos 2 nodos, hay {graph.number_of_nodes()}")
            
        if graph.number_of_nodes() > 5:
             raise AssertionError(f"Demasiados nodos ({graph.number_of_nodes()}) para un dataset simple")
             
        if vals.shape[0] != 20:
             raise AssertionError("Filter values debe tener mismo tamaño que data")
             
        print_test_result("Test Mapper Básico", True)
        print(f"\n{EMOJI['check']} ¡Todos los tests pasaron!")
        
    except Exception as e:
        print_test_result("Error de Ejecución", False, str(e))

def test_takens_embedding(target_func):
    """
    Test para embedding de Takens
    """
    print(f"\n{EMOJI['rocket']} Ejecutando tests para Takens Embedding...")
    
    try:
        # Mock signal: simple sine wave
        t = np.linspace(0, 10, 100)
        signal = np.sin(t)
        
        # 1. Test shape
        # n=100, dim=3, delay=1
        # m = 100 - (3-1)*1 = 98
        emb = target_func(signal, delay=1, dimension=3)
        
        if emb.shape != (98, 3):
            raise AssertionError(f"Forma incorrecta. Esperada (98, 3), obtenida {emb.shape}")
            
        # 2. Test reconstruction logic
        # emb[0] should be [signal[0], signal[1], signal[2]]
        expected_0 = signal[0:3]
        if not np.allclose(emb[0], expected_0):
            raise AssertionError("Lógica de embedding incorrecta en primera fila")
            
        print_test_result("Test Takens Básico", True)
        print(f"\n{EMOJI['check']} ¡Todos los tests pasaron!")
        
    except Exception as e:
        print_test_result("Error de Ejecución", False, str(e))

def test_signal_features(target_func):
    """
    Test para extracción de características TDA de señales
    """
    print(f"\n{EMOJI['rocket']} Ejecutando tests para Features de Señal...")
    
    try:
        # Mock signal
        signal = np.random.randn(200)
        
        features = target_func(signal, fs=100)
        
        required = ['n_cycles', 'max_persistence', 'mean_persistence']
        for k in required:
            if k not in features:
                raise AssertionError(f"Falta característica '{k}'")
                
        if features['max_persistence'] < 0:
            raise AssertionError("Persistencia máxima no puede ser negativa")
            
        print_test_result("Test Extracción Features", True)
        print(f"\n{EMOJI['check']} ¡Todos los tests pasaron!")
        
    except Exception as e:
        print_test_result("Error de Ejecución", False, str(e))

def test_epilepsy_pipeline(target_gen, target_prep, target_feats):
    """
    Test para pipeline completo de epilepsia
    """
    print(f"\n{EMOJI['rocket']} Ejecutando tests para Pipeline de Epilepsia...")
    
    try:
        # 1. Test Generation
        # 23 channels, 1 second, 256 Hz = 256 samples
        eeg, t = target_gen(duration=1.0, fs=256, n_channels=23, state='interictal')
        
        if eeg.shape != (23, 256):
            raise AssertionError(f"Generación incorrecta. Esperado (23, 256), obtenido {eeg.shape}")
            
        # 2. Test Preprocessing
        eeg_prep = target_prep(eeg, fs=256)
        if eeg_prep.shape != (23, 256):
            raise AssertionError(f"Preprocesamiento cambió la forma. Obtenido {eeg_prep.shape}")
            
        # Check normalization (mean approx 0)
        if abs(np.mean(eeg_prep)) > 0.1:
            raise AssertionError("Datos no parecen estar centrados/normalizados correctamente")
            
        # 3. Test Feature Extraction
        # Pass a single channel or formatted segment as expected
        # The function expects a full segment (n_channels, n_samples)
        feats = target_feats(eeg, fs=256)
        
        required_prefixes = ['tda_', 'spectral_', 'temporal_']
        for prefix in required_prefixes:
            found = any(k.startswith(prefix) for k in feats.keys())
            if not found:
                raise AssertionError(f"Faltan características de tipo '{prefix}'")
                
        print_test_result("Test Pipeline Epilepsia", True)
        print(f"\n{EMOJI['check']} ¡Todos los tests pasaron!")
        
    except Exception as e:
        print_test_result("Error de Ejecución", False, str(e))
