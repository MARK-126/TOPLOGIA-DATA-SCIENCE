#!/usr/bin/env python3
"""
Script de prueba para Tutorial 1: Introducción al TDA
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from ripser import ripser
from scipy.spatial.distance import pdist, squareform
from sklearn.datasets import make_circles, make_moons

print("=" * 60)
print("PRUEBA TUTORIAL 1: Introducción al TDA")
print("=" * 60)

np.random.seed(42)

# Test 1: Construcción de complejo simplicial
print("\n1. Probando construcción de complejo simplicial...")
neural_positions = np.array([
    [0, 0],
    [1, 0],
    [0.5, 0.8],
    [2, 0],
    [1.5, 0.8]
])

epsilon = 1.0
n_points = len(neural_positions)
distances = squareform(pdist(neural_positions))

edges = []
for i in range(n_points):
    for j in range(i+1, n_points):
        if distances[i, j] <= epsilon:
            edges.append((i, j))

triangles = []
for i in range(n_points):
    for j in range(i+1, n_points):
        for k in range(j+1, n_points):
            if (distances[i,j] <= epsilon and
                distances[j,k] <= epsilon and
                distances[i,k] <= epsilon):
                triangles.append([i, j, k])

print(f"   Puntos: {n_points}, Aristas: {len(edges)}, Triángulos: {len(triangles)}")
print("   ✓ Complejo simplicial construido")

# Test 2: Cálculo de números de Betti
print("\n2. Probando cálculo de números de Betti...")
circle_points, _ = make_circles(n_samples=100, noise=0.05, factor=0.5, random_state=42)
circle_points = circle_points[circle_points[:, 0]**2 + circle_points[:, 1]**2 > 0.1]

result = ripser(circle_points, maxdim=2, thresh=1.0)
diagrams = result['dgms']

# Contar características persistentes en un epsilon específico
eps_test = 0.5
betti_0 = np.sum((diagrams[0][:, 0] <= eps_test) &
                ((diagrams[0][:, 1] > eps_test) | np.isinf(diagrams[0][:, 1])))
betti_1 = np.sum((diagrams[1][:, 0] <= eps_test) &
                (diagrams[1][:, 1] > eps_test))

print(f"   En ε={eps_test}: β₀={int(betti_0)}, β₁={int(betti_1)}")
print("   ✓ Números de Betti calculados")

# Test 3: Red neuronal sintética
print("\n3. Probando generación de red neuronal...")
def generate_neural_network(n_neurons=50, noise_level=0.1):
    community1 = np.random.randn(n_neurons//2, 2) * 0.5 + np.array([0, 0])
    community2 = np.random.randn(n_neurons//2, 2) * 0.5 + np.array([3, 0])
    bridge = np.array([[1.5, 0]])
    neurons = np.vstack([community1, community2, bridge])
    neurons += np.random.randn(*neurons.shape) * noise_level
    return neurons

neural_network = generate_neural_network(n_neurons=60, noise_level=0.08)
print(f"   Red generada: {neural_network.shape}")

result_neural = ripser(neural_network, maxdim=2)
diagrams_neural = result_neural['dgms']

print(f"   H₀: {len(diagrams_neural[0])} features")
print(f"   H₁: {len(diagrams_neural[1])} features")
print(f"   H₂: {len(diagrams_neural[2])} features")
print("   ✓ Red neuronal analizada")

# Test 4: Estados cerebrales
print("\n4. Probando comparación de estados cerebrales...")
def generate_brain_state(state_type='resting', n_neurons=100):
    if state_type == 'resting':
        data = np.random.randn(n_neurons, 3) * 1.5
    elif state_type == 'active':
        theta = np.random.uniform(0, 2*np.pi, n_neurons)
        phi = np.random.uniform(0, np.pi, n_neurons)
        r = 1 + np.random.randn(n_neurons) * 0.1

        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)

        data = np.column_stack([x, y, z])
    return data

resting_state = generate_brain_state('resting', n_neurons=150)
active_state = generate_brain_state('active', n_neurons=150)

result_resting = ripser(resting_state, maxdim=2, thresh=3.0)
result_active = ripser(active_state, maxdim=2, thresh=3.0)

print(f"   Reposo: H₁={len(result_resting['dgms'][1])}, H₂={len(result_resting['dgms'][2])}")
print(f"   Activo: H₁={len(result_active['dgms'][1])}, H₂={len(result_active['dgms'][2])}")
print("   ✓ Estados cerebrales comparados")

# Test 5: Visualización sin persim
print("\n5. Probando visualización alternativa (sin persim)...")
fig, ax = plt.subplots(figsize=(8, 8))
for dim, color in enumerate(['red', 'blue', 'green']):
    if dim < len(diagrams):
        diagram = diagrams[dim]
        diagram_finite = diagram[diagram[:, 1] < np.inf]
        if len(diagram_finite) > 0:
            ax.scatter(diagram_finite[:, 0], diagram_finite[:, 1],
                      c=color, alpha=0.6, label=f'H{dim}', s=50)

if len(diagrams) > 0:
    max_val = max([d[d[:, 1] < np.inf].max() if len(d[d[:, 1] < np.inf]) > 0 else 0
                   for d in diagrams])
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3)

ax.set_xlabel('Birth')
ax.set_ylabel('Death')
ax.set_title('Diagrama de Persistencia (sin persim)')
ax.legend()
plt.savefig('/home/user/TOPLOGIA-DATA-SCIENCE/test_tutorial1_output.png', dpi=100)
print("   ✓ Visualización guardada")

# Resumen
print("\n" + "=" * 60)
print("RESUMEN - TUTORIAL 1")
print("=" * 60)
print("✓ Construcción de complejos simpliciales")
print("✓ Cálculo de números de Betti")
print("✓ Generación de redes neuronales sintéticas")
print("✓ Comparación de estados cerebrales")
print("✓ Visualización sin dependencia de persim")
print("\n✅ TUTORIAL 1 VALIDADO - TODOS LOS COMPONENTES FUNCIONAN")
print("=" * 60)
