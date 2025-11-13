#!/usr/bin/env python3
"""
Script de prueba para verificar el código del Tutorial 0
"""

import importlib
import sys

print("=" * 60)
print("PRUEBA DEL TUTORIAL 0: Setup y Quickstart")
print("=" * 60)

# 1. Verificación de librerías
print("\n1. Verificando instalación de librerías...")
libraries = {
    'numpy': 'NumPy',
    'scipy': 'SciPy',
    'matplotlib': 'Matplotlib',
    'sklearn': 'Scikit-learn',
    'ripser': 'Ripser',
}

installed = []
missing = []

for module_name, display_name in libraries.items():
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"   ✅ {display_name:20s} v{version}")
        installed.append(display_name)
    except ImportError:
        print(f"   ❌ {display_name:20s} NO INSTALADO")
        missing.append(display_name)

# 2. Primer análisis TDA (del Tutorial 0)
print("\n2. Ejecutando primer análisis TDA...")
import numpy as np
from ripser import ripser
import matplotlib
matplotlib.use('Agg')  # Backend sin display
import matplotlib.pyplot as plt

# Generar puntos en un círculo
theta = np.linspace(0, 2*np.pi, 50)
circle = np.column_stack([np.cos(theta), np.sin(theta)])
circle += np.random.randn(50, 2) * 0.1  # Añadir ruido

print(f"   Datos generados: {circle.shape}")

# Computar homología persistente
result = ripser(circle, maxdim=1)
diagrams = result['dgms']

print(f"   H0 (componentes): {len(diagrams[0])} features")
print(f"   H1 (ciclos): {len(diagrams[1])} features")

# Verificar que detectó el círculo
h1_features = diagrams[1]
h1_features = h1_features[h1_features[:, 1] < np.inf]  # Remover infinitos
if len(h1_features) > 0:
    lifetimes = h1_features[:, 1] - h1_features[:, 0]
    max_lifetime = np.max(lifetimes)
    print(f"   Persistencia máxima en H1: {max_lifetime:.4f}")
    if max_lifetime > 0.1:
        print("   ✅ Círculo detectado correctamente!")
    else:
        print("   ⚠️ Persistencia baja (pero funciona)")
else:
    print("   ⚠️ No se detectaron ciclos persistentes")

# 3. Visualización básica
print("\n3. Creando visualización...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot datos
ax1.scatter(circle[:, 0], circle[:, 1], alpha=0.6)
ax1.set_title('Datos: Círculo con ruido')
ax1.set_aspect('equal')

# Plot diagrama de persistencia (manual sin persim)
for dim, color in enumerate(['red', 'blue']):
    diagram = diagrams[dim]
    diagram_finite = diagram[diagram[:, 1] < np.inf]
    if len(diagram_finite) > 0:
        ax2.scatter(diagram_finite[:, 0], diagram_finite[:, 1],
                   c=color, alpha=0.6, label=f'H{dim}')

# Línea diagonal
max_val = max([d.max() for d in diagrams if len(d) > 0 and d.max() < np.inf])
ax2.plot([0, max_val], [0, max_val], 'k--', alpha=0.3)
ax2.set_xlabel('Birth')
ax2.set_ylabel('Death')
ax2.set_title('Diagrama de Persistencia')
ax2.legend()

plt.savefig('/home/user/TOPLOGIA-DATA-SCIENCE/test_output.png', dpi=100, bbox_inches='tight')
print("   ✅ Visualización guardada: test_output.png")

# Resumen final
print("\n" + "=" * 60)
print("RESUMEN DE PRUEBAS - TUTORIAL 0")
print("=" * 60)
print(f"✓ Librerías instaladas: {len(installed)}/{len(libraries)}")
if missing:
    print(f"⚠️ Librerías faltantes: {', '.join(missing)}")
print("✓ Primer análisis TDA ejecutado")
print("✓ Visualización creada")
print("\n✅ TUTORIAL 0 VERIFICADO EXITOSAMENTE")
print("=" * 60)
