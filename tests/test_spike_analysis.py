"""
Tests para módulo de análisis de spikes.
"""

import pytest
import numpy as np
import sys
sys.path.insert(0, '../src')

from src.neuro_utils.spike_analysis import (
    generate_spike_train,
    spikes_to_state_space,
    compute_correlation_matrix,
    generate_brain_network
)


def test_generate_spike_train_random():
    """Test de generación de spike train aleatorio."""
    spikes = generate_spike_train(n_neurons=10, duration=1000,
                                  firing_rate=5.0, pattern='random')

    assert spikes.shape == (10, 1000)
    assert spikes.min() >= 0
    assert np.sum(spikes) > 0  # Al menos algunos spikes


def test_generate_spike_train_synchronized():
    """Test de spike train sincronizado."""
    spikes = generate_spike_train(n_neurons=10, duration=1000,
                                  pattern='synchronized')

    assert spikes.shape == (10, 1000)

    # Verificar que hay correlación alta entre neuronas
    corr = np.corrcoef(spikes)
    off_diagonal = corr[np.triu_indices(10, k=1)]
    assert np.mean(off_diagonal) > 0.3  # Correlación significativa


def test_spikes_to_state_space():
    """Test de conversión a espacio de estados."""
    spikes = np.random.poisson(0.01, size=(5, 500))

    state_space = spikes_to_state_space(spikes, bin_size=50, stride=25)

    assert state_space.shape[1] == 5  # n_neurons
    assert state_space.shape[0] > 0  # n_bins
    assert state_space.min() >= 0


def test_compute_correlation_matrix():
    """Test de cálculo de matriz de correlación."""
    spikes = np.random.randn(10, 1000)

    corr = compute_correlation_matrix(spikes)

    assert corr.shape == (10, 10)
    assert np.allclose(corr, corr.T)  # Simétrica
    assert np.allclose(np.diag(corr), 1.0)  # Diagonal = 1


def test_generate_brain_network():
    """Test de generación de red cerebral."""
    positions = generate_brain_network(n_neurons=30, n_communities=3)

    assert positions.shape == (30, 2)
    assert np.isfinite(positions).all()


if __name__ == '__main__':
    pytest.main([__file__])
