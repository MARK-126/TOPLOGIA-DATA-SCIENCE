"""
Herramientas para análisis de datos neuronales.

Este módulo proporciona funciones para trabajar con spike trains,
matrices de conectividad, y otros datos neurocientíficos.
"""

import numpy as np
from scipy.stats import poisson
from typing import Tuple, Optional


def generate_spike_train(n_neurons: int,
                        duration: int,
                        firing_rate: float = 5.0,
                        pattern: str = 'random',
                        **kwargs) -> np.ndarray:
    """
    Genera spike trains sintéticos con diferentes patrones.

    Parameters:
    -----------
    n_neurons : int
        Número de neuronas
    duration : int
        Duración en milisegundos
    firing_rate : float
        Tasa de disparo en Hz
    pattern : str
        'random', 'synchronized', 'sequential', 'bursting'
    **kwargs : argumentos adicionales para patrones específicos

    Returns:
    --------
    np.ndarray : Matriz de spikes (n_neurons x duration)
    """
    spikes = np.zeros((n_neurons, duration))

    if pattern == 'random':
        # Spikes independientes de Poisson
        for i in range(n_neurons):
            spikes[i] = poisson.rvs(firing_rate/1000, size=duration)

    elif pattern == 'synchronized':
        # Actividad sincronizada
        sync_prob = kwargs.get('sync_prob', 0.8)
        common = poisson.rvs(firing_rate/1000, size=duration)
        for i in range(n_neurons):
            spikes[i] = common * (np.random.rand(duration) < sync_prob)

    elif pattern == 'sequential':
        # Actividad en secuencia (onda)
        wave_period = kwargs.get('wave_period', 50)
        for t in range(duration):
            active_neuron = (t // wave_period) % n_neurons
            spikes[active_neuron, t] = poisson.rvs(firing_rate*2/1000)

    elif pattern == 'bursting':
        # Bursts periódicos
        burst_freq = kwargs.get('burst_freq', 0.1)  # Hz
        burst_duration = kwargs.get('burst_duration', 50)  # ms

        t = 0
        while t < duration:
            if np.random.rand() < burst_freq * 0.001:
                # Burst
                t_end = min(t + burst_duration, duration)
                for i in range(n_neurons):
                    spikes[i, t:t_end] = poisson.rvs(firing_rate*5/1000,
                                                     size=t_end-t)
                t = t_end
            else:
                # Actividad basal
                for i in range(n_neurons):
                    spikes[i, t] = poisson.rvs(firing_rate*0.1/1000)
                t += 1

    return spikes


def spikes_to_state_space(spikes: np.ndarray,
                         bin_size: int = 50,
                         stride: int = 25) -> np.ndarray:
    """
    Convierte spike trains a representación de espacio de estados.

    Parameters:
    -----------
    spikes : np.ndarray
        Matriz de spikes (n_neurons x time)
    bin_size : int
        Tamaño de la ventana en ms
    stride : int
        Paso de la ventana deslizante

    Returns:
    --------
    np.ndarray : Matriz de estados (n_bins x n_neurons)
    """
    n_neurons, duration = spikes.shape
    n_bins = (duration - bin_size) // stride + 1

    state_space = np.zeros((n_bins, n_neurons))

    for i in range(n_bins):
        start = i * stride
        end = start + bin_size
        state_space[i] = np.sum(spikes[:, start:end], axis=1)

    return state_space


def compute_correlation_matrix(spikes: np.ndarray,
                               window: Optional[int] = None) -> np.ndarray:
    """
    Calcula matriz de correlación entre neuronas.

    Parameters:
    -----------
    spikes : np.ndarray
        Matriz de spikes (n_neurons x time)
    window : int, optional
        Si se proporciona, usa ventana deslizante

    Returns:
    --------
    np.ndarray : Matriz de correlación (n_neurons x n_neurons)
    """
    if window is None:
        # Correlación global
        return np.corrcoef(spikes)
    else:
        # Correlación en ventanas (promedio)
        n_neurons, duration = spikes.shape
        n_windows = duration // window

        corr_matrices = []
        for i in range(n_windows):
            start = i * window
            end = start + window
            corr_matrices.append(np.corrcoef(spikes[:, start:end]))

        return np.mean(corr_matrices, axis=0)


def generate_brain_network(n_neurons: int,
                          n_communities: int = 3,
                          connectivity: float = 0.3,
                          noise: float = 0.1) -> np.ndarray:
    """
    Genera una red neuronal sintética con estructura de comunidades.

    Parameters:
    -----------
    n_neurons : int
        Número total de neuronas
    n_communities : int
        Número de comunidades
    connectivity : float
        Nivel de conectividad (0-1)
    noise : float
        Nivel de ruido espacial

    Returns:
    --------
    np.ndarray : Posiciones de neuronas en espacio 2D
    """
    neurons_per_community = n_neurons // n_communities
    positions = []

    for i in range(n_communities):
        # Centro de cada comunidad
        angle = 2 * np.pi * i / n_communities
        center = 3 * np.array([np.cos(angle), np.sin(angle)])

        # Generar neuronas alrededor del centro
        community = np.random.randn(neurons_per_community, 2) * 0.5 + center
        positions.append(community)

    # Agregar neuronas restantes
    remainder = n_neurons - neurons_per_community * n_communities
    if remainder > 0:
        positions.append(np.random.randn(remainder, 2) * 0.5)

    # Combinar y agregar ruido
    all_positions = np.vstack(positions)
    all_positions += np.random.randn(*all_positions.shape) * noise

    return all_positions


def load_connectivity_matrix(filepath: str) -> np.ndarray:
    """
    Carga matriz de conectividad desde archivo.

    Parameters:
    -----------
    filepath : str
        Ruta al archivo (.npy, .csv, etc.)

    Returns:
    --------
    np.ndarray : Matriz de conectividad
    """
    if filepath.endswith('.npy'):
        return np.load(filepath)
    elif filepath.endswith('.csv'):
        return np.loadtxt(filepath, delimiter=',')
    else:
        raise ValueError(f"Formato de archivo no soportado: {filepath}")
