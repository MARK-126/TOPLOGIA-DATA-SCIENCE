"""
Funciones de visualización para TDA aplicado a neurociencias.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from persim import plot_diagrams
from typing import List, Optional, Tuple


def plot_persistence_diagram(diagrams: List[np.ndarray],
                            title: str = "Diagrama de Persistencia",
                            figsize: Tuple[int, int] = (8, 6)):
    """
    Visualiza un diagrama de persistencia.

    Parameters:
    -----------
    diagrams : list of np.ndarray
        Lista de diagramas de persistencia
    title : str
        Título del gráfico
    figsize : tuple
        Tamaño de la figura
    """
    fig, ax = plt.subplots(figsize=figsize)
    plot_diagrams(diagrams, ax=ax)
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_betti_curves(epsilons: np.ndarray,
                     betti_numbers: List[np.ndarray],
                     labels: Optional[List[str]] = None,
                     title: str = "Números de Betti"):
    """
    Visualiza la evolución de números de Betti.

    Parameters:
    -----------
    epsilons : np.ndarray
        Valores de epsilon
    betti_numbers : list of np.ndarray
        Lista de arrays con números de Betti para cada dimensión
    labels : list of str, optional
        Etiquetas para cada dimensión
    title : str
        Título del gráfico
    """
    n_dims = len(betti_numbers)
    if labels is None:
        labels = [f'β_{i}' for i in range(n_dims)]

    fig, axes = plt.subplots(1, n_dims, figsize=(6*n_dims, 5))

    if n_dims == 1:
        axes = [axes]

    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']

    for i, (betti, label, color) in enumerate(zip(betti_numbers, labels, colors)):
        axes[i].plot(epsilons, betti, linewidth=3, color=color)
        axes[i].fill_between(epsilons, betti, alpha=0.3, color=color)
        axes[i].set_xlabel('Radio ε', fontsize=12)
        axes[i].set_ylabel(label, fontsize=12)
        axes[i].set_title(f'Dimensión {i}', fontsize=12, fontweight='bold')
        axes[i].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()


def plot_spike_raster(spikes: np.ndarray,
                     title: str = "Spike Raster",
                     figsize: Tuple[int, int] = (12, 6)):
    """
    Visualiza spike trains como raster plot.

    Parameters:
    -----------
    spikes : np.ndarray
        Matriz de spikes (n_neurons x time)
    title : str
        Título del gráfico
    figsize : tuple
        Tamaño de la figura
    """
    fig, ax = plt.subplots(figsize=figsize)

    n_neurons, duration = spikes.shape

    for neuron in range(n_neurons):
        spike_times = np.where(spikes[neuron] > 0)[0]
        ax.scatter(spike_times, [neuron]*len(spike_times),
                  c='black', s=2, marker='|')

    ax.set_xlabel('Tiempo (ms)', fontsize=12)
    ax.set_ylabel('Neurona #', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlim([0, duration])
    ax.set_ylim([-1, n_neurons])
    plt.tight_layout()
    plt.show()


def plot_brain_network_2d(positions: np.ndarray,
                          connectivity: Optional[np.ndarray] = None,
                          title: str = "Red Neuronal",
                          figsize: Tuple[int, int] = (10, 8)):
    """
    Visualiza una red neuronal en 2D.

    Parameters:
    -----------
    positions : np.ndarray
        Posiciones de neuronas (n_neurons x 2)
    connectivity : np.ndarray, optional
        Matriz de conectividad
    title : str
        Título del gráfico
    figsize : tuple
        Tamaño de la figura
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Dibujar conexiones si están disponibles
    if connectivity is not None:
        n_neurons = len(positions)
        threshold = np.percentile(connectivity[connectivity > 0], 75)
        for i in range(n_neurons):
            for j in range(i+1, n_neurons):
                if connectivity[i, j] > threshold:
                    ax.plot([positions[i, 0], positions[j, 0]],
                           [positions[i, 1], positions[j, 1]],
                           'gray', alpha=0.3, linewidth=0.5)

    # Dibujar neuronas
    ax.scatter(positions[:, 0], positions[:, 1],
              c='blue', s=100, alpha=0.6, edgecolors='black')

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_distance_matrix(matrix: np.ndarray,
                        labels: Optional[List[str]] = None,
                        title: str = "Matriz de Distancia",
                        cmap: str = 'YlOrRd'):
    """
    Visualiza una matriz de distancias como heatmap.

    Parameters:
    -----------
    matrix : np.ndarray
        Matriz de distancia
    labels : list of str, optional
        Etiquetas para ejes
    title : str
        Título del gráfico
    cmap : str
        Colormap de matplotlib
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(matrix, cmap=cmap, aspect='auto')

    if labels is not None:
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45)
        ax.set_yticklabels(labels)

    # Agregar valores en las celdas
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            text = ax.text(j, i, f'{matrix[i, j]:.2f}',
                          ha="center", va="center",
                          color="black", fontweight='bold')

    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()
