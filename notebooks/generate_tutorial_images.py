"""
Script para generar imágenes explicativas para los tutoriales de TDA
Crea diagramas estáticos de alta calidad para conceptos clave
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch
from matplotlib.collections import LineCollection
from scipy.spatial.distance import pdist, squareform
from ripser import ripser
import os

# Crear directorio de imágenes si no existe
output_dir = '/home/user/TOPLOGIA-DATA-SCIENCE/notebooks/images'
os.makedirs(output_dir, exist_ok=True)

print("🎨 Generando imágenes explicativas...")
print(f"📁 Directorio: {output_dir}\n")

# ==============================================================================
# 1. CONSTRUCCIÓN DE COMPLEJO SIMPLICIAL (Paso a Paso)
# ==============================================================================

def generate_simplicial_construction():
    """Genera imagen mostrando construcción paso a paso de complejo"""
    print("📊 Generando: construcción de complejo simplicial...")

    # Puntos de ejemplo
    points = np.array([
        [0, 0],
        [1, 0],
        [0.5, 0.8],
        [2, 0],
        [1.5, 0.8]
    ])

    epsilon_values = [0.5, 1.0, 1.5]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for idx, eps in enumerate(epsilon_values):
        ax = axes[idx]

        # Calcular distancias
        distances = squareform(pdist(points))

        # Encontrar aristas
        edges = []
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                if distances[i, j] <= eps:
                    edges.append((i, j))

        # Encontrar triángulos
        triangles = []
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                for k in range(j+1, len(points)):
                    if (distances[i,j] <= eps and
                        distances[j,k] <= eps and
                        distances[i,k] <= eps):
                        triangles.append([i, j, k])

        # Dibujar triángulos (rellenos)
        for tri in triangles:
            triangle_points = points[tri]
            ax.fill(triangle_points[:, 0], triangle_points[:, 1],
                   alpha=0.3, color='#2196f3', edgecolor='none')

        # Dibujar aristas
        for i, j in edges:
            ax.plot([points[i, 0], points[j, 0]],
                   [points[i, 1], points[j, 1]],
                   'gray', linewidth=2, alpha=0.7, zorder=1)

        # Dibujar puntos
        ax.scatter(points[:, 0], points[:, 1],
                  c='#f44336', s=300, zorder=3, edgecolors='black', linewidths=2)

        # Etiquetas
        for i, point in enumerate(points):
            ax.annotate(f'{i}', xy=point, xytext=(0, 0),
                       textcoords='offset points', fontsize=14,
                       fontweight='bold', ha='center', va='center', color='white')

        # Título y estadísticas
        ax.set_title(f'ε = {eps:.1f}\nAristas: {len(edges)} | Triángulos: {len(triangles)}',
                    fontsize=14, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(-0.3, 1.2)

    plt.suptitle('Construcción de Complejo Simplicial (Vietoris-Rips)\n' +
                'Mayor ε → Más Conexiones → Estructura Más Densa',
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/simplicial_construction_steps.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✅ Guardado: simplicial_construction_steps.png")


# ==============================================================================
# 2. ANATOMÍA DE DIAGRAMA DE PERSISTENCIA
# ==============================================================================

def generate_persistence_diagram_anatomy():
    """Genera imagen explicando partes de un diagrama de persistencia"""
    print("\n📊 Generando: anatomía de diagrama de persistencia...")

    fig, ax = plt.subplots(figsize=(12, 12))

    # Crear diagrama sintético
    h0_points = np.array([
        [0.1, 0.3],
        [0.2, 0.5],
        [0.15, 0.8],
        [0.0, np.inf]
    ])

    h1_points = np.array([
        [0.3, 0.9],
        [0.5, 1.2],
        [0.7, 0.85]
    ])

    # Plotear H0 (rojo)
    h0_finite = h0_points[h0_points[:, 1] < np.inf]
    ax.scatter(h0_finite[:, 0], h0_finite[:, 1],
              c='#f44336', s=200, alpha=0.7, label='H₀ (Componentes)',
              edgecolors='black', linewidths=2, zorder=3)

    # Plotear H1 (azul)
    ax.scatter(h1_points[:, 0], h1_points[:, 1],
              c='#2196f3', s=200, alpha=0.7, label='H₁ (Ciclos)',
              edgecolors='black', linewidths=2, zorder=3)

    # Línea diagonal
    ax.plot([0, 1.5], [0, 1.5], 'k--', linewidth=3, alpha=0.3, label='Diagonal')

    # Anotaciones explicativas
    # Feature persistente (lejos de diagonal)
    feature_persistent = h1_points[0]
    ax.annotate('Feature PERSISTENTE\n(lejos de diagonal)\n→ Importante',
               xy=feature_persistent, xytext=(feature_persistent[0]-0.3, feature_persistent[1]+0.3),
               fontsize=12, fontweight='bold', color='#2196f3',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='#e3f2fd', edgecolor='#2196f3', linewidth=2),
               arrowprops=dict(arrowstyle='->', lw=2, color='#2196f3'))

    # Feature de ruido (cerca de diagonal)
    feature_noise = h1_points[2]
    ax.annotate('Feature de RUIDO\n(cerca de diagonal)\n→ Poco importante',
               xy=feature_noise, xytext=(feature_noise[0]+0.2, feature_noise[1]-0.3),
               fontsize=12, fontweight='bold', color='#ff9800',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='#fff3e0', edgecolor='#ff9800', linewidth=2),
               arrowprops=dict(arrowstyle='->', lw=2, color='#ff9800'))

    # Explicar birth/death
    example_point = h1_points[1]
    birth_x, death_y = example_point
    ax.plot([birth_x, birth_x], [0, death_y], 'g--', linewidth=2, alpha=0.5)
    ax.plot([0, birth_x], [death_y, death_y], 'r--', linewidth=2, alpha=0.5)

    ax.annotate('Birth (nacimiento)\nε donde aparece',
               xy=(birth_x, 0), xytext=(birth_x+0.15, -0.15),
               fontsize=11, color='green',
               bbox=dict(boxstyle='round', facecolor='#e8f5e9', alpha=0.8),
               arrowprops=dict(arrowstyle='->', color='green'))

    ax.annotate('Death (muerte)\nε donde desaparece',
               xy=(0, death_y), xytext=(-0.3, death_y+0.1),
               fontsize=11, color='red',
               bbox=dict(boxstyle='round', facecolor='#ffebee', alpha=0.8),
               arrowprops=dict(arrowstyle='->', color='red'))

    # Persistence (lifetime)
    mid_x = birth_x + 0.05
    mid_y = death_y - 0.05
    ax.annotate('', xy=(mid_x, mid_y), xytext=(mid_x-0.15, mid_y+0.15),
               arrowprops=dict(arrowstyle='<->', lw=3, color='purple'))
    ax.text(mid_x-0.08, mid_y+0.2, 'Persistencia\n(lifetime)',
           fontsize=11, fontweight='bold', color='purple',
           bbox=dict(boxstyle='round', facecolor='#f3e5f5', alpha=0.8))

    # Configuración de ejes
    ax.set_xlabel('Birth (Nacimiento)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Death (Muerte)', fontsize=14, fontweight='bold')
    ax.set_title('Anatomía de un Diagrama de Persistencia\n' +
                'Regla de Oro: Puntos lejos de la diagonal = Características importantes',
                fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.3, 1.5)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/persistence_diagram_anatomy.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✅ Guardado: persistence_diagram_anatomy.png")


# ==============================================================================
# 3. EVOLUCIÓN DE NÚMEROS DE BETTI
# ==============================================================================

def generate_betti_evolution():
    """Genera imagen mostrando evolución de números de Betti"""
    print("\n📊 Generando: evolución de números de Betti...")

    # Generar círculo
    theta = np.linspace(0, 2*np.pi, 80)
    circle = np.column_stack([np.cos(theta), np.sin(theta)])
    circle += np.random.randn(80, 2) * 0.05

    # Calcular homología
    result = ripser(circle, maxdim=2, thresh=1.5)
    diagrams = result['dgms']

    # Calcular betti numbers para diferentes epsilon
    epsilons = np.linspace(0.01, 1.5, 100)
    betti_0 = np.zeros(100)
    betti_1 = np.zeros(100)

    for i, eps in enumerate(epsilons):
        betti_0[i] = np.sum((diagrams[0][:, 0] <= eps) &
                           ((diagrams[0][:, 1] > eps) | np.isinf(diagrams[0][:, 1])))
        betti_1[i] = np.sum((diagrams[1][:, 0] <= eps) &
                           (diagrams[1][:, 1] > eps))

    # Crear figura con 3 paneles
    fig = plt.figure(figsize=(18, 5))

    # Panel 1: Datos
    ax1 = plt.subplot(131)
    ax1.scatter(circle[:, 0], circle[:, 1], c='#2196f3', s=50, alpha=0.6, edgecolors='black')
    ax1.set_title('Datos: Círculo con Ruido', fontsize=14, fontweight='bold')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')

    # Panel 2: β₀ (componentes)
    ax2 = plt.subplot(132)
    ax2.plot(epsilons, betti_0, linewidth=3, color='#f44336')
    ax2.fill_between(epsilons, betti_0, alpha=0.3, color='#f44336')
    ax2.set_xlabel('Radio ε', fontsize=12, fontweight='bold')
    ax2.set_ylabel('β₀ (Componentes)', fontsize=12, fontweight='bold')
    ax2.set_title('Dimensión 0: Componentes Conectadas\nβ₀ converge a 1', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=1, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Convergencia')
    ax2.legend()

    # Panel 3: β₁ (ciclos)
    ax3 = plt.subplot(133)
    ax3.plot(epsilons, betti_1, linewidth=3, color='#2196f3')
    ax3.fill_between(epsilons, betti_1, alpha=0.3, color='#2196f3')
    ax3.set_xlabel('Radio ε', fontsize=12, fontweight='bold')
    ax3.set_ylabel('β₁ (Ciclos)', fontsize=12, fontweight='bold')
    ax3.set_title('Dimensión 1: Ciclos\nDetecta el círculo (β₁ = 1)', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # Marcar región donde β₁ = 1
    indices_one = np.where(betti_1 == 1)[0]
    if len(indices_one) > 0:
        start_eps = epsilons[indices_one[0]]
        end_eps = epsilons[indices_one[-1]]
        ax3.axvspan(start_eps, end_eps, alpha=0.2, color='green', label='Círculo detectado')
        ax3.legend()

    plt.suptitle('Evolución de Números de Betti vs. Radio ε\nCómo cambia la topología al aumentar la escala',
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/betti_numbers_evolution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✅ Guardado: betti_numbers_evolution.png")


# ==============================================================================
# 4. COMPARACIÓN H₀, H₁, H₂
# ==============================================================================

def generate_homology_dimensions_comparison():
    """Genera imagen comparando diferentes dimensiones de homología"""
    print("\n📊 Generando: comparación de dimensiones homológicas...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # H₀ - Componentes conectadas
    ax = axes[0]
    # Dibujar 3 componentes separadas
    comp1 = np.array([[0, 0], [0.3, 0], [0.15, 0.2]])
    comp2 = np.array([[1, 0], [1.3, 0], [1.15, 0.2]])
    comp3 = np.array([[2, 0], [2.3, 0], [2.15, 0.2]])

    for comp, color in [(comp1, '#f44336'), (comp2, '#4caf50'), (comp3, '#2196f3')]:
        ax.scatter(comp[:, 0], comp[:, 1], s=300, c=color, edgecolors='black', linewidths=2)
        # Conectar puntos dentro del componente
        for i in range(len(comp)):
            for j in range(i+1, len(comp)):
                ax.plot([comp[i, 0], comp[j, 0]], [comp[i, 1], comp[j, 1]],
                       color=color, linewidth=2, alpha=0.5)

    ax.set_title('H₀: Componentes Conectadas\nβ₀ = 3 (tres componentes)',
                fontsize=14, fontweight='bold')
    ax.text(1.15, -0.4, 'Cuenta partes\ndesconectadas', fontsize=12, ha='center',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    ax.set_xlim(-0.3, 2.6)
    ax.set_ylim(-0.6, 0.5)
    ax.set_aspect('equal')
    ax.axis('off')

    # H₁ - Ciclos
    ax = axes[1]
    theta = np.linspace(0, 2*np.pi, 50)
    circle = np.column_stack([np.cos(theta), np.sin(theta)])
    ax.plot(circle[:, 0], circle[:, 1], 'o-', color='#2196f3', linewidth=3, markersize=8)
    ax.fill(circle[:, 0], circle[:, 1], alpha=0.2, color='#2196f3')
    ax.annotate('', xy=(0.7, 0.7), xytext=(-0.7, -0.7),
               arrowprops=dict(arrowstyle='->', lw=3, color='red'))
    ax.text(0, -1.5, 'Cuenta ciclos\n(loops, huecos)', fontsize=12, ha='center',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    ax.set_title('H₁: Ciclos/Loops\nβ₁ = 1 (un ciclo)',
                fontsize=14, fontweight='bold')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.8, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')

    # H₂ - Cavidades (esfera)
    ax = axes[2]
    # Dibujar representación 2D de esfera
    outer_circle = plt.Circle((0, 0), 1, fill=False, edgecolor='#9c27b0', linewidth=3)
    inner_circle = plt.Circle((0, 0), 0.7, fill=True, facecolor='#f3e5f5', edgecolor='none', alpha=0.5)
    ax.add_patch(inner_circle)
    ax.add_patch(outer_circle)

    # Dibujar algunos puntos en la superficie
    theta_surf = np.linspace(0, 2*np.pi, 20)
    surf_x = np.cos(theta_surf)
    surf_y = np.sin(theta_surf)
    ax.scatter(surf_x, surf_y, s=100, c='#9c27b0', edgecolors='black', linewidths=1.5, zorder=3)

    ax.text(0, 0, 'Cavidad\nvacía', fontsize=14, ha='center', va='center',
           fontweight='bold', color='#9c27b0')
    ax.text(0, -1.5, 'Cuenta cavidades\n(volúmenes cerrados)', fontsize=12, ha='center',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    ax.set_title('H₂: Cavidades\nβ₂ = 1 (una cavidad)',
                fontsize=14, fontweight='bold')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.8, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')

    plt.suptitle('Dimensiones de Homología: ¿Qué Cuenta Cada Una?\n' +
                'H₀ = Componentes | H₁ = Ciclos | H₂ = Cavidades',
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/homology_dimensions_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✅ Guardado: homology_dimensions_comparison.png")


# ==============================================================================
# 5. CONCEPTO DE PERSISTENCIA
# ==============================================================================

def generate_persistence_concept():
    """Genera imagen explicando el concepto de persistencia"""
    print("\n📊 Generando: concepto de persistencia...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Generar datos: círculo con ruido
    np.random.seed(42)
    theta = np.linspace(0, 2*np.pi, 50)
    circle = np.column_stack([np.cos(theta), np.sin(theta)])
    circle += np.random.randn(50, 2) * 0.08

    # Agregar algunos puntos de ruido aleatorio
    noise_points = np.random.randn(10, 2) * 1.5
    all_points = np.vstack([circle, noise_points])

    epsilon_values = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5]

    for idx, eps in enumerate(epsilon_values):
        ax = axes[idx // 3, idx % 3]

        # Calcular distancias
        distances = squareform(pdist(all_points))

        # Encontrar aristas
        edges = []
        for i in range(len(all_points)):
            for j in range(i+1, len(all_points)):
                if distances[i, j] <= eps:
                    edges.append((i, j))

        # Dibujar aristas
        for i, j in edges:
            ax.plot([all_points[i, 0], all_points[j, 0]],
                   [all_points[i, 1], all_points[j, 1]],
                   'gray', linewidth=0.5, alpha=0.3)

        # Dibujar puntos del círculo (azul)
        ax.scatter(circle[:, 0], circle[:, 1], c='#2196f3', s=50,
                  edgecolors='black', linewidths=1, alpha=0.7, zorder=3)

        # Dibujar puntos de ruido (rojo)
        ax.scatter(noise_points[:, 0], noise_points[:, 1], c='#f44336', s=50,
                  edgecolors='black', linewidths=1, alpha=0.7, zorder=3)

        ax.set_title(f'ε = {eps:.1f}', fontsize=13, fontweight='bold')
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)
        ax.axis('off')

        # Agregar anotación según el valor de epsilon
        if eps <= 0.3:
            ax.text(0, -2.2, 'Muchas componentes\n(ruido y señal separados)',
                   fontsize=10, ha='center', bbox=dict(boxstyle='round', facecolor='#ffebee'))
        elif eps < 1.0:
            ax.text(0, -2.2, 'Círculo se forma\n(característica persistente)',
                   fontsize=10, ha='center', bbox=dict(boxstyle='round', facecolor='#e3f2fd'))
        else:
            ax.text(0, -2.2, 'Todo conectado\n(características desaparecen)',
                   fontsize=10, ha='center', bbox=dict(boxstyle='round', facecolor='#fff3e0'))

    plt.suptitle('Concepto de Persistencia: ¿Qué características sobreviven?\n' +
                'Ruido = Aparece y desaparece rápido | Señal = Persiste a través de múltiples escalas',
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/persistence_concept.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✅ Guardado: persistence_concept.png")


# ==============================================================================
# EJECUTAR TODAS LAS GENERACIONES
# ==============================================================================

if __name__ == "__main__":
    generate_simplicial_construction()
    generate_persistence_diagram_anatomy()
    generate_betti_evolution()
    generate_homology_dimensions_comparison()
    generate_persistence_concept()

    print("\n" + "="*70)
    print("🎉 ¡TODAS LAS IMÁGENES GENERADAS EXITOSAMENTE!")
    print("="*70)
    print(f"\n📁 Ubicación: {output_dir}")
    print("\n📊 Imágenes creadas:")
    print("   1. simplicial_construction_steps.png")
    print("   2. persistence_diagram_anatomy.png")
    print("   3. betti_numbers_evolution.png")
    print("   4. homology_dimensions_comparison.png")
    print("   5. persistence_concept.png")
    print("\n✅ Listas para usar en tutoriales!")
