"""
Funciones auxiliares para tutoriales de TDA
Contiene funciones de utilidad para visualización, cálculos y generación de datos
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from ripser import ripser
import time
import pandas as pd
from scipy.stats import poisson
from sklearn.decomposition import PCA
from persim import bottleneck, sliced_wasserstein
import networkx as nx
from scipy.stats import pearsonr
from sklearn.cluster import SpectralClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap
try:
    import gudhi as gd
except ImportError:
    gd = None
try:
    import kmapper as km
except ImportError:
    km = None
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.stats import zscore
import pandas as pd

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

def compare_filtrations(points, max_dim=2):
    """
    Compara diferentes tipos de filtraciones en los mismos datos.
    """
    results = {}
    
    # 1. Vietoris-Rips
    start = time.time()
    rips_result = ripser(points, maxdim=max_dim)
    rips_time = time.time() - start
    results['Vietoris-Rips'] = {
        'diagrams': rips_result['dgms'],
        'time': rips_time
    }
    
    # 2. Alpha complex (si GUDHI está disponible y dim <= 3)
    if gd is not None and points.shape[1] <= 3:
        start = time.time()
        alpha_complex = gd.AlphaComplex(points=points)
        simplex_tree = alpha_complex.create_simplex_tree()
        persistence = simplex_tree.persistence()
        
        alpha_diagrams = [[] for _ in range(max_dim + 1)]
        for dim, (birth, death) in persistence:
            if dim <= max_dim:
                alpha_diagrams[dim].append([birth, death])
        
        alpha_diagrams = [np.array(d) if len(d) > 0 else np.array([]).reshape(0, 2) 
                         for d in alpha_diagrams]
        alpha_time = time.time() - start
        
        results['Alpha'] = {
            'diagrams': alpha_diagrams,
            'time': alpha_time
        }
    
    return results

def generate_brain_state_realistic(state_type, n_neurons=100, noise=0.1):
    """
    Genera estados cerebrales sintéticos con propiedades realistas.
    """
    if state_type == 'sleep':
        base = np.random.randn(n_neurons, 1) @ np.random.randn(1, 5)
        data = base + np.random.randn(n_neurons, 5) * noise
        
    elif state_type == 'wakeful':
        data = np.random.randn(n_neurons, 5) * 1.5
        
    elif state_type == 'attention':
        data = np.zeros((n_neurons, 5))
        data[:n_neurons//3] = np.random.randn(n_neurons//3, 5) * 2.0
        data[n_neurons//3:] = np.random.randn(2*n_neurons//3, 5) * 0.3
        
    elif state_type == 'memory':
        theta = np.linspace(0, 4*np.pi, n_neurons)
        data = np.column_stack([
            np.cos(theta),
            np.sin(theta),
            np.cos(2*theta) * 0.5,
            np.sin(2*theta) * 0.5,
            np.random.randn(n_neurons) * noise
        ])
    
    return data

def generate_spike_trains(n_neurons=20, duration=1000, base_rate=5.0, 
                         correlation=0.3, pattern_type='random'):
    """
    Genera spike trains sintéticos con diferentes patrones.
    """
    spike_trains = np.zeros((n_neurons, duration))
    
    if pattern_type == 'random':
        for i in range(n_neurons):
            spike_trains[i] = poisson.rvs(base_rate/1000, size=duration)
            
    elif pattern_type == 'synchronized':
        common_pattern = poisson.rvs(base_rate/1000, size=duration)
        for i in range(n_neurons):
            spike_trains[i] = common_pattern * (np.random.rand(duration) < 0.8)
            
    elif pattern_type == 'sequential':
        for t in range(duration):
            active_neuron = (t // 20) % n_neurons
            spike_trains[active_neuron, t] = poisson.rvs(base_rate*3/1000)
    
    return spike_trains

def spike_trains_to_state_space(spike_trains, bin_size=50, stride=25):
    """
    Convierte spike trains a representación en espacio de estados.
    """
    n_neurons, duration = spike_trains.shape
    n_bins = (duration - bin_size) // stride + 1
    
    state_space = np.zeros((n_bins, n_neurons))
    
    for i in range(n_bins):
        start = i * stride
        end = start + bin_size
        state_space[i] = np.sum(spike_trains[:, start:end], axis=1)
    
    return state_space

def extract_topological_features(diagram, dim=1):
    """
    Extrae características escalares de un diagrama de persistencia.
    """
    features = {}
    
    if len(diagram) <= dim or len(diagram[dim]) == 0:
        return {'n_features': 0, 'max_persistence': 0, 
                'mean_persistence': 0, 'entropy': 0}
    
    dgm = diagram[dim]
    dgm = dgm[np.isfinite(dgm[:, 1])]
    
    if len(dgm) == 0:
        return {'n_features': 0, 'max_persistence': 0, 
                'mean_persistence': 0, 'entropy': 0}
    
    lifetimes = dgm[:, 1] - dgm[:, 0]
    
    features['n_features'] = len(dgm)
    features['max_persistence'] = np.max(lifetimes)
    features['mean_persistence'] = np.mean(lifetimes)
    features['total_persistence'] = np.sum(lifetimes)
    
    if np.sum(lifetimes) > 0:
        probs = lifetimes / np.sum(lifetimes)
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        features['entropy'] = entropy
    else:
        features['entropy'] = 0
    
    return features

def generate_fmri_timeseries(n_regions=50, n_timepoints=200, 
                             n_communities=3, noise_level=0.3):
    """
    Genera series temporales sintéticas de fMRI con estructura de comunidades.
    """
    regions_per_community = n_regions // n_communities
    
    timeseries = np.zeros((n_regions, n_timepoints))
    labels = np.zeros(n_regions, dtype=int)
    
    t = np.linspace(0, 4*np.pi, n_timepoints)
    global_signal = np.sin(t) * 0.2
    
    for comm in range(n_communities):
        start_idx = comm * regions_per_community
        end_idx = start_idx + regions_per_community if comm < n_communities - 1 else n_regions
        
        common_signal = np.sin(t + comm * np.pi/3) + 0.5 * np.sin(2*t + comm)
        
        for i in range(start_idx, end_idx):
            timeseries[i] = common_signal + np.random.randn(n_timepoints) * noise_level
            labels[i] = comm
            
    timeseries += global_signal
    return timeseries, labels

def compute_functional_connectivity(timeseries):
    """
    Calcula matriz de conectividad funcional (correlación).
    """
    n_regions = timeseries.shape[0]
    conn_matrix = np.zeros((n_regions, n_regions))
    
    for i in range(n_regions):
        for j in range(i, n_regions):
            corr, _ = pearsonr(timeseries[i], timeseries[j])
            conn_matrix[i, j] = corr
            conn_matrix[j, i] = corr
    
    return conn_matrix

def connectivity_to_distance(conn_matrix):
    """
    Convierte matriz de conectividad (correlación) a matriz de distancia.
    """
    dist_matrix = 1 - np.abs(conn_matrix)
    np.fill_diagonal(dist_matrix, 0)
    return dist_matrix

def analyze_connectivity_topology(conn_matrix, maxdim=2):
    """
    Analiza la topología de una matriz de conectividad.
    """
    dist_matrix = connectivity_to_distance(conn_matrix)
    result = ripser(dist_matrix, maxdim=maxdim, distance_matrix=True, thresh=1.0)
    return result['dgms']

def detect_communities_spectral(conn_matrix, n_clusters=3):
    """
    Detecta comunidades usando clustering espectral.
    """
    affinity = np.abs(conn_matrix)
    clustering = SpectralClustering(n_clusters=n_clusters, 
                                   affinity='precomputed',
                                   random_state=42)
    labels = clustering.fit_predict(affinity)
    return labels

def extract_connectivity_features(conn_matrix, maxdim=2):
    """
    Extrae características topológicas y de grafo de una matriz de conectividad.
    """
    features = {}
    
    # 1. Características de grafo tradicionales
    # Usamos valor absoluto para el grafo subyacente
    G = nx.from_numpy_array(np.abs(conn_matrix))
    
    # Algunas métricas pueden ser lentas para grafos muy grandes
    features['avg_clustering'] = nx.average_clustering(G, weight='weight')
    features['avg_degree'] = np.mean([d for n, d in G.degree(weight='weight')])
    features['density'] = nx.density(G)
    
    # 2. Características topológicas
    diagrams = analyze_connectivity_topology(conn_matrix, maxdim=maxdim)
    
    # H1
    features.update(_summarize_diagram(diagrams, 1, 'cycles'))
    # H2
    features.update(_summarize_diagram(diagrams, 2, 'cavities'))
    
    return features

def _summarize_diagram(diagrams, dim, name):
    """Helper interno para resumir diagrama"""
    feats = {}
    if len(diagrams) > dim and len(diagrams[dim]) > 0:
        dgm = diagrams[dim][np.isfinite(diagrams[dim][:, 1])]
        if len(dgm) > 0:
            lifetimes = dgm[:, 1] - dgm[:, 0]
            feats[f'n_{name}'] = len(dgm)
            feats[f'max_{name}_persistence'] = np.max(lifetimes)
            feats[f'mean_{name}_persistence'] = np.mean(lifetimes)
            return feats
            
    feats[f'n_{name}'] = 0
    feats[f'max_{name}_persistence'] = 0
    feats[f'mean_{name}_persistence'] = 0
    return feats

def simple_mapper(data, filter_func, n_intervals=10, overlap=0.3, 
                 clustering=None):
    """
    Implementación didáctica del algoritmo Mapper.
    """
    if clustering is None:
        clustering = DBSCAN(eps=0.5, min_samples=3)
    
    # 1. Aplicar función de filtro
    filter_values = filter_func(data)
    
    # 2. Crear cover
    f_min, f_max = filter_values.min(), filter_values.max()
    interval_length = (f_max - f_min) / (n_intervals * (1 - overlap))
    step = interval_length * (1 - overlap)
    
    intervals = []
    for i in range(n_intervals):
        start = f_min + i * step
        end = start + interval_length
        intervals.append((start, end))
    
    # 3. Clustering en cada intervalo
    nodes = {}  # {node_id: [indices de puntos]}
    node_id = 0
    
    for interval_idx, (start, end) in enumerate(intervals):
        # Puntos en este intervalo
        mask = (filter_values >= start) & (filter_values <= end)
        indices = np.where(mask)[0]
        
        if len(indices) == 0:
            continue
        
        # Clustering
        subset = data[indices]
        # Check if subset is large enough for clustering
        if len(subset) < 1: 
             continue
             
        try:
            labels = clustering.fit_predict(subset)
        except Exception:
            continue
        
        # Crear nodos
        for label in set(labels):
            if label == -1:  # Ruido en DBSCAN
                continue
            cluster_mask = labels == label
            cluster_indices = indices[cluster_mask]
            nodes[node_id] = cluster_indices
            node_id += 1
    
    # 4. Construir grafo (nerve)
    G = nx.Graph()
    
    # Agregar nodos
    for nid in nodes.keys():
        G.add_node(nid, size=len(nodes[nid]))
    
    # Agregar aristas (si comparten puntos)
    node_ids = list(nodes.keys())
    for i, nid1 in enumerate(node_ids):
        for nid2 in node_ids[i+1:]:\n            intersection = set(nodes[nid1]) & set(nodes[nid2])
            if len(intersection) > 0:
                G.add_edge(nid1, nid2, weight=len(intersection))
    
    return G, nodes, filter_values

def pca_filter(data):
    """Filtro PCA (1D)"""
    pca = PCA(n_components=1)
    return pca.fit_transform(data).ravel()

def generate_brain_trajectory(n_timepoints=500, n_neurons=50, 
                             trajectory_type='cyclic'):
    """
    Genera una trayectoria de estados cerebrales en espacio de alta dimensión.
    """
    if trajectory_type == 'cyclic':
        # Ciclo: descanso -> atención -> memoria -> descanso
        t = np.linspace(0, 4*np.pi, n_timepoints)
        
        # Manifold base (círculo en 2D, embebido en n_neurons dimensiones)
        trajectory = np.zeros((n_timepoints, n_neurons))
        trajectory[:, 0] = 3 * np.cos(t)  # Primera dimensión
        trajectory[:, 1] = 3 * np.sin(t)  # Segunda dimensión
        
        # Agregar estructura en dimensiones adicionales
        for i in range(2, min(10, n_neurons)):
            trajectory[:, i] = 0.5 * np.sin(t * (i-1) / 2) * np.cos(t * i / 3)
        
        # Ruido en dimensiones restantes
        if n_neurons > 10:
             trajectory[:, 10:] = np.random.randn(n_timepoints, max(0, n_neurons-10)) * 0.3
        
        # Etiquetas de fase
        phase = (t % (2*np.pi)) / (2*np.pi)
        labels = np.zeros(n_timepoints, dtype=int)
        labels[phase < 0.25] = 0  # Descanso
        labels[(phase >= 0.25) & (phase < 0.5)] = 1  # Atención
        labels[(phase >= 0.5) & (phase < 0.75)] = 2  # Memoria
        labels[phase >= 0.75] = 0  # De vuelta a descanso
        
    elif trajectory_type == 'branching':
        # Bifurcación: estado inicial -> decisión A o B
        trajectory = np.zeros((n_timepoints, n_neurons))
        labels = np.zeros(n_timepoints, dtype=int)
        
        # Fase 1: Común (0-200)
        limit_phase1 = min(200, n_timepoints)
        t1 = np.linspace(0, 2, limit_phase1)
        trajectory[:limit_phase1, 0] = t1
        labels[:limit_phase1] = 0
        
        if n_timepoints > 200:
            # Fase 2: Bifurcación
            mid = n_timepoints // 2
            end_phase2 = min(350, n_timepoints)
            
            # Rama A (200-350)
            if end_phase2 > 200:
                len_phase2 = end_phase2 - 200
                t2 = np.linspace(0, np.pi, len_phase2)
                trajectory[200:end_phase2, 0] = 2 + np.cos(t2)
                trajectory[200:end_phase2, 1] = np.sin(t2)
                labels[200:end_phase2] = 1
            
            # Rama B (350-500)
            if n_timepoints > 350:
                len_phase3 = n_timepoints - 350
                t3 = np.linspace(0, np.pi, len_phase3)
                trajectory[350:, 0] = 2 + np.cos(t3)
                trajectory[350:, 1] = -np.sin(t3)
                labels[350:] = 2
        
        # Ruido
        trajectory += np.random.randn(n_timepoints, n_neurons) * 0.2
    
    return trajectory, labels

def takens_embedding(timeseries, delay=1, dimension=3):
    """
    Crea embedding de Takens de una serie temporal.
    """
    n = len(timeseries)
    m = n - (dimension - 1) * delay
    
    if m <= 0:
        raise ValueError("Serie temporal muy corta para estos parámetros")
    
    embedded = np.zeros((m, dimension))
    
    for i in range(dimension):
        start = i * delay
        end = start + m
        embedded[:, i] = timeseries[start:end]
    
    return embedded

def estimate_delay(timeseries, max_delay=100):
    """
    Estima delay óptimo usando primer mínimo de autocorrelación.
    """
    # Centrar
    ts_centered = timeseries - np.mean(timeseries)
    autocorr = np.correlate(ts_centered, ts_centered, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    autocorr = autocorr / autocorr[0]
    
    # Primer cruce por cero o mínimo local
    for i in range(1, min(max_delay, len(autocorr)-1)):
        if autocorr[i] < autocorr[i-1] and autocorr[i] < autocorr[i+1]:
            return i
    
    # Si no hay mínimo, usar 1/e
    threshold = 1/np.e
    for i in range(1, min(max_delay, len(autocorr))):
        if autocorr[i] < threshold:
            return i
    
    return 1

def generate_eeg_signal(duration=10, fs=250, state='normal'):
    """
    Genera señal EEG sintética.
    """
    n_samples = int(duration * fs)
    t = np.linspace(0, duration, n_samples)
    
    if state == 'normal':
        # Ritmos normales: alpha (8-13 Hz), beta (13-30 Hz)
        alpha = 0.5 * np.sin(2 * np.pi * 10 * t)  # 10 Hz
        beta = 0.3 * np.sin(2 * np.pi * 20 * t)   # 20 Hz
        noise = 0.2 * np.random.randn(n_samples)
        eeg = alpha + beta + noise
        
    elif state == 'seizure':
        # Crisis: spike-wave a 3 Hz, alta amplitud
        spike_wave = 2.0 * np.sin(2 * np.pi * 3 * t)
        harmonics = 0.5 * np.sin(2 * np.pi * 6 * t)
        noise = 0.1 * np.random.randn(n_samples)
        eeg = spike_wave + harmonics + noise
        
    elif state == 'sleep':
        # Sueño: delta (0.5-4 Hz), alta amplitud
        delta = 1.5 * np.sin(2 * np.pi * 2 * t)
        theta = 0.3 * np.sin(2 * np.pi * 6 * t)  # 6 Hz
        noise = 0.15 * np.random.randn(n_samples)
        eeg = delta + theta + noise
    
    return t, eeg

def extract_tda_features_from_signal(signal, fs=250):
    """
    Extrae características topológicas de una señal temporal.
    """
    # Normalizar
    signal_norm = zscore(signal)
    
    # Embedding
    delay = estimate_delay(signal_norm, max_delay=30)
    embedded = takens_embedding(signal_norm, delay=delay, dimension=3)
    
    # TDA
    result = ripser(embedded, maxdim=1, thresh=5.0)
    dgm1 = result['dgms'][1]
    
    # Características
    features = {}
    
    if len(dgm1) > 0:
        dgm1_finite = dgm1[np.isfinite(dgm1[:, 1])]
        if len(dgm1_finite) > 0:
            lifetimes = dgm1_finite[:, 1] - dgm1_finite[:, 0]
            features['n_cycles'] = len(dgm1_finite)
            features['max_persistence'] = np.max(lifetimes)
            features['mean_persistence'] = np.mean(lifetimes)
        else:
             features['n_cycles'] = 0
             features['max_persistence'] = 0
             features['mean_persistence'] = 0
    else:
        features['n_cycles'] = 0
        features['max_persistence'] = 0
        features['mean_persistence'] = 0
    
    return features

def generate_realistic_eeg_segment(duration=10, fs=256, state='interictal', n_channels=23):
    """
    Genera segmento de EEG multicanal realista.
    """
    n_samples = int(duration * fs)
    t = np.linspace(0, duration, n_samples)
    
    eeg_data = np.zeros((n_channels, n_samples))
    
    for ch in range(n_channels):
        if state == 'interictal':
            # Estado normal: múltiples ritmos mezclados
            # Alpha (8-13 Hz)
            alpha = 0.3 * np.sin(2 * np.pi * np.random.uniform(8, 13) * t)
            # Beta (13-30 Hz)
            beta = 0.2 * np.sin(2 * np.pi * np.random.uniform(13, 30) * t)
            # Theta (4-8 Hz)
            theta = 0.15 * np.sin(2 * np.pi * np.random.uniform(4, 8) * t)
            # Ruido fisiológico
            noise = 0.3 * np.random.randn(n_samples)
            
            eeg_data[ch] = alpha + beta + theta + noise
            
        elif state == 'ictal':
            # Crisis: actividad rítmica de alta amplitud
            # Spike-wave típico de epilepsia: 3-5 Hz dominante
            seizure_freq = np.random.uniform(3, 5)
            spike_wave = 2.5 * np.sin(2 * np.pi * seizure_freq * t)
            
            # Armónicos (característica de crisis)
            harmonics = 0.8 * np.sin(2 * np.pi * 2 * seizure_freq * t)
            harmonics += 0.4 * np.sin(2 * np.pi * 3 * seizure_freq * t)
            
            # Actividad de alta frecuencia (HFOs)
            hfo = 0.3 * np.sin(2 * np.pi * np.random.uniform(80, 200) * t)
            
            # Sincronización entre canales (característica clave)
            phase_offset = np.random.uniform(0, 0.2)  # Poca variación
            
            noise = 0.15 * np.random.randn(n_samples)  # Menos ruido
            
            eeg_data[ch] = spike_wave + harmonics + hfo + noise
            eeg_data[ch] = np.roll(eeg_data[ch], int(phase_offset * fs))
    
    return eeg_data, t

def preprocess_eeg(eeg_data, fs=256):
    """
    Pipeline de preprocesamiento profesional para EEG.
    """
    n_channels, n_samples = eeg_data.shape
    
    # 1. Filtro bandpass (0.5-50 Hz)
    nyquist = fs / 2
    low = 0.5 / nyquist
    high = 50 / nyquist
    b, a = signal.butter(4, [low, high], btype='band')
    
    eeg_filtered = np.zeros_like(eeg_data)
    for ch in range(n_channels):
        eeg_filtered[ch] = signal.filtfilt(b, a, eeg_data[ch])
    
    # 2. Notch filter (60 Hz) - Usamos implementación simple de IIR
    b_notch, a_notch = signal.iirnotch(60, 30, fs)
    for ch in range(n_channels):
        eeg_filtered[ch] = signal.filtfilt(b_notch, a_notch, eeg_filtered[ch])
    
    # 3. Common average reference
    car = np.mean(eeg_filtered, axis=0)
    eeg_car = eeg_filtered - car
    
    # 4. Normalización por canal (z-score)
    eeg_normalized = np.zeros_like(eeg_car)
    for ch in range(n_channels):
        eeg_normalized[ch] = zscore(eeg_car[ch])
    
    return eeg_normalized

def extract_comprehensive_features(eeg_segment, fs=256):
    """
    Extrae características completas: TDA + espectrales + temporales.
    """
    features = {}
    
    # Preprocesar
    eeg_prep = preprocess_eeg(eeg_segment, fs)
    
    # Usar primer canal (en práctica: todos los canales)
    signal_data = eeg_prep[0] # Renombramos para evitar conflicto de nombres
    
    # === CARACTERÍSTICAS TDA ===
    delay = estimate_delay(signal_data) # Usamos la función ya definida en utils
    embedded = takens_embedding(signal_data, delay=delay, dimension=3)
    
    # Subsampling para eficiencia
    if len(embedded) > 500:
        indices = np.random.choice(len(embedded), 500, replace=False)
        embedded = embedded[indices]
    
    result = ripser(embedded, maxdim=1, thresh=5.0)
    dgm1 = result['dgms'][1]
    
    if len(dgm1) > 0:
        dgm1_finite = dgm1[np.isfinite(dgm1[:, 1])]
        if len(dgm1_finite) > 0:
            lifetimes = dgm1_finite[:, 1] - dgm1_finite[:, 0]
            features['tda_n_cycles'] = len(dgm1_finite)
            features['tda_max_persistence'] = np.max(lifetimes)
            features['tda_mean_persistence'] = np.mean(lifetimes)
            features['tda_std_persistence'] = np.std(lifetimes)
            features['tda_total_persistence'] = np.sum(lifetimes)
            features['tda_p50_persistence'] = np.percentile(lifetimes, 50)
            features['tda_p90_persistence'] = np.percentile(lifetimes, 90)
        else:
            for k in ['tda_n_cycles', 'tda_max_persistence', 'tda_mean_persistence',
                     'tda_std_persistence', 'tda_total_persistence', 
                     'tda_p50_persistence', 'tda_p90_persistence']:
                features[k] = 0
    else:
        for k in ['tda_n_cycles', 'tda_max_persistence', 'tda_mean_persistence',
                 'tda_std_persistence', 'tda_total_persistence',
                 'tda_p50_persistence', 'tda_p90_persistence']:
            features[k] = 0
    
    # === CARACTERÍSTICAS ESPECTRALES ===
    freqs = fftfreq(len(signal_data), 1/fs)
    fft_vals = np.abs(fft(signal_data))
    
    # Bandas de frecuencia
    delta = (freqs >= 0.5) & (freqs <= 4)
    theta = (freqs >= 4) & (freqs <= 8)
    alpha = (freqs >= 8) & (freqs <= 13)
    beta = (freqs >= 13) & (freqs <= 30)
    gamma = (freqs >= 30) & (freqs <= 50)
    
    features['spectral_delta_power'] = np.sum(fft_vals[delta])
    features['spectral_theta_power'] = np.sum(fft_vals[theta])
    features['spectral_alpha_power'] = np.sum(fft_vals[alpha])
    features['spectral_beta_power'] = np.sum(fft_vals[beta])
    features['spectral_gamma_power'] = np.sum(fft_vals[gamma])
    
    # Ratios
    total_power = np.sum(fft_vals[freqs >= 0])
    features['spectral_delta_ratio'] = features['spectral_delta_power'] / (total_power + 1e-10)
    features['spectral_theta_alpha_ratio'] = features['spectral_theta_power'] / (features['spectral_alpha_power'] + 1e-10)
    
    # Frecuencia dominante (asegurar rango positivo)
    pos_mask = (freqs >= 0) & (freqs <= 50)
    if np.any(pos_mask):
        dominant_idx = np.argmax(fft_vals[pos_mask])
        features['spectral_dominant_freq'] = freqs[pos_mask][dominant_idx]
    else:
        features['spectral_dominant_freq'] = 0
    
    # === CARACTERÍSTICAS TEMPORALES ===
    features['temporal_mean'] = np.mean(signal_data)
    features['temporal_std'] = np.std(signal_data)
    features['temporal_skewness'] = float(pd.Series(signal_data).skew())
    features['temporal_kurtosis'] = float(pd.Series(signal_data).kurtosis())
    features['temporal_rms'] = np.sqrt(np.mean(signal_data**2))
    
    # Zero-crossings
    features['temporal_zero_crossings'] = np.sum(np.diff(np.sign(signal_data)) != 0)
    
    return features
