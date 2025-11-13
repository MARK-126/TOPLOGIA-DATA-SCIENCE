"""
Funciones auxiliares para tutoriales de TDA
Contiene funciones de utilidad para visualización, cálculos y generación de datos
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial.distance import pdist, squareform
from ripser import ripser


def plot_persistence_diagram_manual(diagrams, title="Diagrama de Persistencia", ax=None):
    """
    Visualiza diagrama de persistencia sin dependencia de persim

    Arguments:
    diagrams -- lista de diagramas de persistencia de ripser
    title -- título del gráfico
    ax -- axes de matplotlib (opcional)

    Returns:
    ax -- axes modificado
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    colors = ['red', 'blue', 'green', 'purple']

    for dim, color in enumerate(colors):
        if dim < len(diagrams):
            diagram = diagrams[dim]
            diagram_finite = diagram[diagram[:, 1] < np.inf]

            if len(diagram_finite) > 0:
                ax.scatter(diagram_finite[:, 0], diagram_finite[:, 1],
                          c=color, alpha=0.6, label=f'H{dim}', s=50, edgecolors='black')

    # Línea diagonal
    if len(diagrams) > 0:
        max_val = max([d[d[:, 1] < np.inf].max() if len(d[d[:, 1] < np.inf]) > 0 else 0
                       for d in diagrams])
        ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, linewidth=2)

    ax.set_xlabel('Birth (Nacimiento)', fontsize=12)
    ax.set_ylabel('Death (Muerte)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    return ax


def plot_betti_curves(epsilons, betti_0, betti_1, betti_2, title="Curvas de Betti"):
    """
    Visualiza la evolución de los números de Betti

    Arguments:
    epsilons -- array de valores de epsilon
    betti_0, betti_1, betti_2 -- arrays de números de Betti
    title -- título del gráfico
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # β₀
    axes[0].plot(epsilons, betti_0, linewidth=3, color='#e74c3c')
    axes[0].fill_between(epsilons, betti_0, alpha=0.3, color='#e74c3c')
    axes[0].set_xlabel('Radio ε', fontsize=12)
    axes[0].set_ylabel('β₀ (Componentes)', fontsize=12)
    axes[0].set_title('Dimensión 0: Componentes Conectadas', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # β₁
    axes[1].plot(epsilons, betti_1, linewidth=3, color='#3498db')
    axes[1].fill_between(epsilons, betti_1, alpha=0.3, color='#3498db')
    axes[1].set_xlabel('Radio ε', fontsize=12)
    axes[1].set_ylabel('β₁ (Ciclos)', fontsize=12)
    axes[1].set_title('Dimensión 1: Ciclos/Loops', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    # β₂
    axes[2].plot(epsilons, betti_2, linewidth=3, color='#2ecc71')
    axes[2].fill_between(epsilons, betti_2, alpha=0.3, color='#2ecc71')
    axes[2].set_xlabel('Radio ε', fontsize=12)
    axes[2].set_ylabel('β₂ (Cavidades)', fontsize=12)
    axes[2].set_title('Dimensión 2: Cavidades', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()


def visualize_simplicial_complex_simple(points, edges, triangles, epsilon, title="Complejo Simplicial"):
    """
    Visualización simple de complejo simplicial sin NetworkX

    Arguments:
    points -- array de puntos (n_points, 2)
    edges -- lista de tuplas (i, j)
    triangles -- lista de listas [i, j, k]
    epsilon -- radio usado
    title -- título del gráfico
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Dibujar triángulos (sombreados)
    for tri in triangles:
        triangle_points = points[tri]
        ax.fill(triangle_points[:, 0], triangle_points[:, 1],
                alpha=0.2, color='lightblue', edgecolor='none')

    # Dibujar aristas
    for i, j in edges:
        ax.plot([points[i, 0], points[j, 0]],
                [points[i, 1], points[j, 1]],
                'gray', linewidth=1.5, alpha=0.6, zorder=1)

    # Dibujar puntos
    ax.scatter(points[:, 0], points[:, 1],
               c='red', s=200, zorder=3, edgecolors='black', linewidths=2)

    # Etiquetas
    for i, point in enumerate(points):
        ax.annotate(f'{i}', xy=point, xytext=(5, 5),
                    textcoords='offset points', fontsize=12, fontweight='bold')

    ax.set_title(f"{title}\n(ε = {epsilon:.2f}, Aristas: {len(edges)}, Triángulos: {len(triangles)})",
                 fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def compare_persistence_diagrams(diagrams_list, labels, title="Comparación de Diagramas"):
    """
    Compara múltiples diagramas de persistencia en un solo plot

    Arguments:
    diagrams_list -- lista de listas de diagramas
    labels -- lista de etiquetas para cada conjunto
    title -- título del gráfico
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    colors = ['red', 'blue', 'green', 'purple', 'orange']
    markers = ['o', 's', '^', 'D', 'v']

    for idx, (diagrams, label) in enumerate(zip(diagrams_list, labels)):
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]

        # Solo plotear H1 para simplicidad
        if len(diagrams) > 1:
            diagram = diagrams[1]
            diagram_finite = diagram[diagram[:, 1] < np.inf]

            if len(diagram_finite) > 0:
                ax.scatter(diagram_finite[:, 0], diagram_finite[:, 1],
                          c=color, alpha=0.6, label=label, s=80,
                          marker=marker, edgecolors='black')

    # Línea diagonal
    max_val = 2.0
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, linewidth=2)

    ax.set_xlabel('Birth', fontsize=12)
    ax.set_ylabel('Death', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def print_section_header(title, level=1):
    """
    Imprime un encabezado de sección con formato

    Arguments:
    title -- título de la sección
    level -- nivel del encabezado (1, 2, 3)
    """
    chars = {1: '=', 2: '-', 3: '.'}
    char = chars.get(level, '=')
    width = 70

    print('\n' + char * width)
    print(f' {title}')
    print(char * width + '\n')


def print_test_result(test_name, passed=True, message=""):
    """
    Imprime resultado de un test con formato

    Arguments:
    test_name -- nombre del test
    passed -- True si pasó, False si falló
    message -- mensaje adicional
    """
    status = "\033[92m✅ PASÓ\033[0m" if passed else "\033[91m❌ FALLÓ\033[0m"
    print(f"{test_name}: {status}")
    if message:
        print(f"   {message}")


def create_test_cases_tutorial1():
    """
    Crea casos de prueba predefinidos para Tutorial 1

    Returns:
    dictionary con casos de prueba
    """
    # Red neuronal simple
    neural_positions = np.array([
        [0, 0],
        [1, 0],
        [0.5, 0.8],
        [2, 0],
        [1.5, 0.8]
    ])

    # Puntos en círculo
    theta = np.linspace(0, 2*np.pi, 50)
    circle = np.column_stack([np.cos(theta), np.sin(theta)])
    circle += np.random.randn(50, 2) * 0.1

    return {
        'neural_positions': neural_positions,
        'circle_points': circle
    }


def load_sample_eeg_data(n_channels=23, duration=10, fs=256):
    """
    Genera datos EEG sintéticos de ejemplo

    Arguments:
    n_channels -- número de canales
    duration -- duración en segundos
    fs -- frecuencia de muestreo

    Returns:
    eeg_data -- array (n_channels, n_samples)
    """
    n_samples = int(duration * fs)
    t = np.linspace(0, duration, n_samples)
    eeg_data = np.zeros((n_channels, n_samples))

    for ch in range(n_channels):
        # Ritmos cerebrales
        delta = 0.8 * np.sin(2 * np.pi * np.random.uniform(1, 4) * t)
        theta = 0.6 * np.sin(2 * np.pi * np.random.uniform(4, 8) * t)
        alpha = 1.2 * np.sin(2 * np.pi * np.random.uniform(8, 13) * t)
        beta = 0.4 * np.sin(2 * np.pi * np.random.uniform(13, 30) * t)
        noise = 0.5 * np.random.randn(n_samples)

        eeg_data[ch, :] = delta + theta + alpha + beta + noise

    return eeg_data


# Constantes útiles
COLORS = {
    'primary': '#2196f3',
    'success': '#4caf50',
    'warning': '#ff9800',
    'danger': '#f44336',
    'info': '#00bcd4'
}

EMOJI = {
    'check': '✅',
    'cross': '❌',
    'warning': '⚠️',
    'star': '⭐',
    'brain': '🧠',
    'chart': '📊',
    'fire': '🔥',
    'rocket': '🚀'
}
