"""
Tests para módulo de persistencia.
"""

import pytest
import numpy as np
import sys
sys.path.insert(0, '../src')

from src.tda_tools.persistence import (
    compute_persistence,
    extract_features,
    compute_betti_numbers,
    filter_noise,
    compare_diagrams
)


def test_compute_persistence_rips():
    """Test básico de cálculo de persistencia con Rips."""
    # Generar datos simples (círculo)
    theta = np.linspace(0, 2*np.pi, 20)
    points = np.column_stack([np.cos(theta), np.sin(theta)])

    result = compute_persistence(points, maxdim=1, method='rips')

    assert 'diagrams' in result
    assert 'method' in result
    assert len(result['diagrams']) >= 2
    assert result['method'] == 'Vietoris-Rips'


def test_extract_features():
    """Test de extracción de características."""
    # Diagrama sintético
    diagram = np.array([[0, 0.5], [0.1, 0.8], [0.2, 1.5]])

    features = extract_features(diagram, dim=0)

    assert 'n_features' in features
    assert 'max_persistence' in features
    assert 'entropy' in features
    assert features['n_features'] == 3
    assert features['max_persistence'] > 0


def test_extract_features_empty():
    """Test con diagrama vacío."""
    diagram = np.array([]).reshape(0, 2)

    features = extract_features(diagram, dim=0)

    assert features['n_features'] == 0
    assert features['max_persistence'] == 0


def test_compute_betti_numbers():
    """Test de cálculo de números de Betti."""
    # Diagrama sintético
    diagrams = [
        np.array([[0, 0.5], [0, 1.0], [0, np.inf]]),  # H0
        np.array([[0.3, 0.8], [0.4, 1.2]])  # H1
    ]

    betti = compute_betti_numbers(diagrams, epsilon=0.6)

    assert len(betti) == 2
    assert betti[0] >= 1  # Al menos una componente
    assert betti[1] >= 0  # Ciclos


def test_filter_noise():
    """Test de filtrado de ruido."""
    # Diagrama con ruido y señal
    diagram = np.array([
        [0, 0.05],  # Ruido
        [0, 0.08],  # Ruido
        [0.1, 0.8],  # Señal
        [0.2, 1.5]   # Señal
    ])

    filtered = filter_noise(diagram, min_persistence=0.1)

    assert len(filtered) < len(diagram)
    assert len(filtered) >= 2  # Al menos las 2 características con señal


def test_compare_diagrams():
    """Test de comparación de diagramas."""
    dgm1 = np.array([[0, 0.5], [0.1, 0.8]])
    dgm2 = np.array([[0, 0.6], [0.15, 0.85]])

    dist_bottleneck = compare_diagrams(dgm1, dgm2, metric='bottleneck')
    dist_wasserstein = compare_diagrams(dgm1, dgm2, metric='wasserstein')

    assert dist_bottleneck >= 0
    assert dist_wasserstein >= 0
    assert dist_bottleneck < np.inf


if __name__ == '__main__':
    pytest.main([__file__])
