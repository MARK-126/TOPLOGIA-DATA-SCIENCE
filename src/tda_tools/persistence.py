"""
Herramientas para análisis de homología persistente.

Este módulo proporciona funciones de alto nivel para calcular y analizar
homología persistente en datos neurocientíficos.
"""

import numpy as np
from ripser import ripser
import gudhi as gd
from typing import Tuple, List, Optional, Dict
import warnings


def compute_persistence(data: np.ndarray,
                       maxdim: int = 2,
                       thresh: float = np.inf,
                       method: str = 'rips') -> Dict:
    """
    Calcula homología persistente usando diferentes métodos.

    Parameters:
    -----------
    data : np.ndarray
        Matriz de puntos (n_points x n_dimensions) o matriz de distancias
    maxdim : int
        Dimensión máxima de homología a calcular
    thresh : float
        Umbral máximo para la filtración
    method : str
        'rips' para Vietoris-Rips, 'alpha' para Alpha complex

    Returns:
    --------
    dict : Diccionario con diagramas de persistencia y metadatos
    """
    if method == 'rips':
        result = ripser(data, maxdim=maxdim, thresh=thresh)
        return {
            'diagrams': result['dgms'],
            'method': 'Vietoris-Rips',
            'maxdim': maxdim
        }

    elif method == 'alpha':
        if data.shape[1] > 3:
            warnings.warn("Alpha complex es más eficiente en dimensión ≤3. Usando Rips.")
            return compute_persistence(data, maxdim, thresh, method='rips')

        alpha_complex = gd.AlphaComplex(points=data)
        simplex_tree = alpha_complex.create_simplex_tree()
        persistence = simplex_tree.persistence()

        # Convertir a formato estándar
        diagrams = [[] for _ in range(maxdim + 1)]
        for dim, (birth, death) in persistence:
            if dim <= maxdim:
                diagrams[dim].append([birth, death])

        diagrams = [np.array(d) if len(d) > 0 else np.array([]).reshape(0, 2)
                   for d in diagrams]

        return {
            'diagrams': diagrams,
            'method': 'Alpha',
            'maxdim': maxdim
        }

    else:
        raise ValueError(f"Método desconocido: {method}")


def extract_features(diagram: np.ndarray, dim: int = 1) -> Dict[str, float]:
    """
    Extrae características escalares de un diagrama de persistencia.

    Parameters:
    -----------
    diagram : np.ndarray
        Diagrama de persistencia (diagrams[dim])
    dim : int
        Dimensión homológica

    Returns:
    --------
    dict : Diccionario con características extraídas
    """
    if len(diagram) == 0:
        return {
            'n_features': 0,
            'max_persistence': 0,
            'mean_persistence': 0,
            'std_persistence': 0,
            'total_persistence': 0,
            'entropy': 0
        }

    # Filtrar infinitos
    dgm = diagram[np.isfinite(diagram[:, 1])]

    if len(dgm) == 0:
        return {
            'n_features': 0,
            'max_persistence': 0,
            'mean_persistence': 0,
            'std_persistence': 0,
            'total_persistence': 0,
            'entropy': 0
        }

    # Lifetimes
    lifetimes = dgm[:, 1] - dgm[:, 0]

    # Características básicas
    features = {
        'n_features': len(dgm),
        'max_persistence': np.max(lifetimes),
        'mean_persistence': np.mean(lifetimes),
        'std_persistence': np.std(lifetimes),
        'total_persistence': np.sum(lifetimes)
    }

    # Entropía de persistencia
    if np.sum(lifetimes) > 0:
        probs = lifetimes / np.sum(lifetimes)
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        features['entropy'] = entropy
    else:
        features['entropy'] = 0

    return features


def compute_betti_numbers(diagrams: List[np.ndarray],
                         epsilon: float) -> List[int]:
    """
    Calcula números de Betti en un valor específico de epsilon.

    Parameters:
    -----------
    diagrams : list of np.ndarray
        Lista de diagramas de persistencia
    epsilon : float
        Valor de escala para evaluar

    Returns:
    --------
    list of int : Números de Betti [β₀, β₁, β₂, ...]
    """
    betti_numbers = []

    for dim, dgm in enumerate(diagrams):
        if len(dgm) == 0:
            betti_numbers.append(0)
            continue

        # Contar características que existen en epsilon
        count = np.sum((dgm[:, 0] <= epsilon) &
                      ((dgm[:, 1] > epsilon) | np.isinf(dgm[:, 1])))
        betti_numbers.append(int(count))

    return betti_numbers


def filter_noise(diagram: np.ndarray,
                 min_persistence: float = 0.1) -> np.ndarray:
    """
    Filtra características con baja persistencia (ruido).

    Parameters:
    -----------
    diagram : np.ndarray
        Diagrama de persistencia
    min_persistence : float
        Persistencia mínima para considerar una característica como señal

    Returns:
    --------
    np.ndarray : Diagrama filtrado
    """
    if len(diagram) == 0:
        return diagram

    lifetimes = diagram[:, 1] - diagram[:, 0]
    mask = (lifetimes >= min_persistence) | np.isinf(diagram[:, 1])

    return diagram[mask]


def compare_diagrams(dgm1: np.ndarray, dgm2: np.ndarray,
                    metric: str = 'bottleneck') -> float:
    """
    Calcula distancia entre dos diagramas de persistencia.

    Parameters:
    -----------
    dgm1, dgm2 : np.ndarray
        Diagramas de persistencia a comparar
    metric : str
        'bottleneck' o 'wasserstein'

    Returns:
    --------
    float : Distancia entre diagramas
    """
    from persim import bottleneck, sliced_wasserstein

    if len(dgm1) == 0 or len(dgm2) == 0:
        warnings.warn("Uno de los diagramas está vacío")
        return np.inf

    if metric == 'bottleneck':
        return bottleneck(dgm1, dgm2)
    elif metric == 'wasserstein':
        return sliced_wasserstein(dgm1, dgm2)
    else:
        raise ValueError(f"Métrica desconocida: {metric}")
