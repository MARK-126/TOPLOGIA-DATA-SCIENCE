"""
Script para crear Tutorial 2 v2 con formato interactivo
"""

import nbformat as nbf

# Crear notebook
nb = nbf.v4.new_notebook()

# Lista de celdas
cells = []

# ================ CELDA 0: Título y objetivos ================
cells.append(nbf.v4.new_markdown_cell("""# Tutorial 2: Homología Persistente Avanzada

## Aplicaciones a Patrones Neuronales (Versión Interactiva)

**Autor:** MARK-126
**Nivel:** Intermedio-Avanzado
**Tiempo estimado:** 120-150 minutos

---

## 🎯 Objetivos de Aprendizaje

Al completar este tutorial, serás capaz de:

1. ✅ Dominar diferentes tipos de filtraciones (Rips, Alpha, Čech)
2. ✅ Calcular distancias entre diagramas de persistencia
3. ✅ Aplicar TDA a spike trains neuronales
4. ✅ Optimizar cálculos para datasets grandes
5. ✅ Interpretar resultados en contexto neurobiológico

---

## ⚠️ Nota Importante sobre Ejercicios

Este notebook contiene **4 ejercicios interactivos** donde deberás completar código.

- Los ejercicios están marcados con `# YOUR CODE STARTS HERE` y `# YOUR CODE ENDS HERE`
- Después de cada ejercicio hay un **test automático** para verificar tu implementación
- Si ves `✅ Todos los tests pasaron`, ¡lo hiciste bien!
- Si ves `❌ Error`, revisa tu código y vuelve a intentar

---"""))

# ================ CELDA 1: Tabla de contenidos ================
cells.append(nbf.v4.new_markdown_cell("""<a name='toc'></a>
## 📚 Tabla de Contenidos

- [1 - Setup e Importaciones](#1)
- [2 - Tipos de Filtraciones](#2)
- [3 - Distancias entre Diagramas](#3)
- [4 - Aplicación: Estados Cerebrales](#4)
    - [Ejercicio 1 - generate_brain_state_realistic](#ex-1)
- [5 - Spike Trains Neuronales](#5)
    - [Ejercicio 2 - generate_spike_trains](#ex-2)
    - [Ejercicio 3 - spike_trains_to_state_space](#ex-3)
- [6 - Características Topológicas para ML](#6)
    - [Ejercicio 4 - extract_topological_features](#ex-4)
- [7 - Optimización y Resumen](#7)

---"""))

# ================ CELDA 2: Setup ================
cells.append(nbf.v4.new_markdown_cell("""<a name='1'></a>
## 1 - Setup e Importaciones

[Volver al índice](#toc)"""))

cells.append(nbf.v4.new_code_cell("""# Importaciones estándar
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# TDA avanzado
from ripser import ripser
from persim import plot_diagrams, bottleneck, sliced_wasserstein
from scipy.spatial.distance import pdist, squareform
from scipy.stats import poisson
from sklearn.decomposition import PCA
import pandas as pd

# Módulos locales
from tda_tests import (
    test_generate_brain_state_realistic,
    test_generate_spike_trains,
    test_spike_trains_to_state_space,
    test_extract_topological_features_tutorial2
)

# Configuración
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("notebook", font_scale=1.2)
np.random.seed(42)

print("✅ Bibliotecas importadas correctamente")
print(f"✅ NumPy version: {np.__version__}")"""))

# ================ CELDA 3: Tipos de filtraciones (teoría) ================
cells.append(nbf.v4.new_markdown_cell("""<a name='2'></a>
## 2 - Tipos de Filtraciones

[Volver al índice](#toc)

### 2.1 ¿Qué es una filtración?

Una **filtración** es una secuencia de complejos simpliciales anidados:
$$\\emptyset = K_0 \\subseteq K_1 \\subseteq K_2 \\subseteq \\ldots \\subseteq K_n = K$$

Cada tipo de filtración tiene **ventajas específicas** para diferentes tipos de datos.

### 2.2 Comparación de Filtraciones

| Filtración | Ventaja | Desventaja | Aplicación Neural |
|-----------|---------|------------|-------------------|
| **Vietoris-Rips** | Rápido, simple | Puede añadir simplejos espurios | Análisis de conectividad general |
| **Alpha** | Geométricamente preciso | Solo para baja dimensión | Visualización de subredes |
| **Filtración de grafo** | Natural para redes | Requiere estructura de grafo | Conectomas cerebrales |

**Para datos neuronales de alta dimensión, Vietoris-Rips es la elección práctica.**

---"""))

# ================ CELDA 4: Distancias entre diagramas (teoría) ================
cells.append(nbf.v4.new_markdown_cell("""<a name='3'></a>
## 3 - Distancias entre Diagramas de Persistencia

[Volver al índice](#toc)

### 3.1 ¿Por qué necesitamos distancias?

Para comparar estados cerebrales, necesitamos **cuantificar diferencias** entre sus topologías.

### 3.2 Principales Métricas

#### A. Distancia de Bottleneck
$$d_B(D_1, D_2) = \\inf_{\\gamma} \\sup_{p \\in D_1} \\|p - \\gamma(p)\\|_\\infty$$

**Interpretación:** Peor caso de emparejamiento entre puntos
- **Robusta** a outliers
- Mide la **máxima diferencia** entre características

#### B. Distancia de Wasserstein (Sliced)
Aproximación rápida de Wasserstein mediante proyecciones 1D.
- **Sensible** a todas las características
- Mide **diferencia global**

---"""))

# ================ CELDA 5: Ejercicio 1 intro ================
cells.append(nbf.v4.new_markdown_cell("""<a name='4'></a>
## 4 - Aplicación: Comparación de Estados Cerebrales

[Volver al índice](#toc)

<a name='ex-1'></a>
### Ejercicio 1 - generate_brain_state_realistic

Implementa la generación de estados cerebrales realistas con diferentes propiedades topológicas.

**Instrucciones:**
1. **Sleep:** Activación sincronizada, baja dimensionalidad (proyección 1D)
2. **Wakeful:** Activación dispersa, alta dimensionalidad (ruido gaussiano)
3. **Attention:** Subredes focales activas (subconjunto activo)
4. **Memory:** Patrones cíclicos (estructura sinusoidal)

**Pasos:**
- Para cada estado, generar matriz (n_neurons x 5)
- Usar propiedades matemáticas específicas para cada estado"""))

# ================ CELDA 6: Ejercicio 1 código ================
cells.append(nbf.v4.new_code_cell("""# EJERCICIO 1: Generar Estados Cerebrales Realistas

def generate_brain_state_realistic(state_type, n_neurons=100, noise=0.1):
    \"\"\"
    Genera estados cerebrales sintéticos con propiedades realistas.

    Arguments:
    state_type -- 'sleep', 'wakeful', 'attention', 'memory'
    n_neurons -- número de neuronas
    noise -- nivel de ruido

    Returns:
    data -- array (n_neurons, 5) con activaciones en espacio de estados
    \"\"\"
    if state_type == 'sleep':
        # Sueño: activación sincronizada, baja dimensionalidad
        # Proyectar desde 1D a 5D: base @ random_projection + noise
        # (approx. 2 lines)
        # YOUR CODE STARTS HERE


        # YOUR CODE ENDS HERE

    elif state_type == 'wakeful':
        # Vigilia: activación dispersa, alta dimensionalidad
        # Simplemente ruido gaussiano en 5D
        # (approx. 1 line)
        # YOUR CODE STARTS HERE

        # YOUR CODE ENDS HERE

    elif state_type == 'attention':
        # Atención: subredes focales activas
        # Primera tercera parte muy activa, resto con actividad basal
        # (approx. 4 lines)
        # YOUR CODE STARTS HERE




        # YOUR CODE ENDS HERE

    elif state_type == 'memory':
        # Memoria: patrones cíclicos (bucles de retroalimentación)
        # Usar funciones seno/coseno con theta lineal
        # (approx. 6 lines)
        # YOUR CODE STARTS HERE






        # YOUR CODE ENDS HERE

    return data"""))

# ================ CELDA 7: Test ejercicio 1 ================
cells.append(nbf.v4.new_code_cell("""# Test del Ejercicio 1
print("🧠 Generando estados cerebrales sintéticos...\\n")

states = {
    'Sueño': generate_brain_state_realistic('sleep', n_neurons=80),
    'Vigilia': generate_brain_state_realistic('wakeful', n_neurons=80),
    'Atención': generate_brain_state_realistic('attention', n_neurons=80),
    'Memoria': generate_brain_state_realistic('memory', n_neurons=80)
}

for name, data in states.items():
    print(f"   {name}: {data.shape}")

# Test automático
test_generate_brain_state_realistic(generate_brain_state_realistic)"""))

# ================ CELDA 8: Visualización estados + análisis topológico ================
cells.append(nbf.v4.new_code_cell("""# Calcular persistencia para cada estado
print("\\n⏳ Calculando homología persistente...\\n")
diagrams = {}

for name, data in states.items():
    result = ripser(data, maxdim=1, thresh=3.0)
    diagrams[name] = result['dgms']
    print(f"✓ {name}: H₀={len(result['dgms'][0])}, H₁={len(result['dgms'][1])}")

# Visualizar diagramas
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for idx, (name, dgm) in enumerate(diagrams.items()):
    plot_diagrams(dgm, ax=axes[idx])
    axes[idx].set_title(f'{name}\\nH₁={len(dgm[1])} ciclos',
                       fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()

print("\\n✅ Análisis topológico completado")"""))

# ================ CELDA 9: Caja de resumen ================
cells.append(nbf.v4.new_markdown_cell("""<div style="background-color:#e3f2fd; padding:15px; border-left:5px solid #2196f3; margin: 20px 0;">

**💡 Lo que debes recordar:**

- Diferentes **estados cerebrales** tienen **firmas topológicas** distintivas
- **Sleep**: Baja dimensionalidad → pocos ciclos (H₁ bajo)
- **Memory**: Estructura cíclica → muchos ciclos (H₁ alto)
- **Distancias entre diagramas** cuantifican similitud entre estados
- **Bottleneck** es robusta, **Wasserstein** es sensible globalmente

</div>

---"""))

# ================ CELDA 10: Spike trains teoría ================
cells.append(nbf.v4.new_markdown_cell("""<a name='5'></a>
## 5 - Spike Trains Neuronales

[Volver al índice](#toc)

### 5.1 ¿Qué son los Spike Trains?

Los **spike trains** son secuencias de potenciales de acción (spikes) de neuronas:
```
Neurona 1: |-----|---------|---|-----|--------|
Neurona 2: |---|-----|----------|--|----------|
Neurona 3: |--------|---|-----|---------------|--|
           0   100   200   300   400   500   600 ms
```

### 5.2 Construcción del Espacio de Estados

Para aplicar TDA:
1. **Ventana deslizante:** Dividir en bins temporales
2. **Vector de activación:** Contar spikes por neurona en cada bin
3. **Espacio de estados:** Cada bin = punto en espacio de dimensión N
4. **Analizar topología** de la trayectoria

---

<a name='ex-2'></a>
### Ejercicio 2 - generate_spike_trains

Implementa la generación de spike trains con diferentes patrones de actividad.

**Instrucciones:**
1. **Random:** Spikes independientes (Poisson)
2. **Synchronized:** Todas las neuronas disparan juntas
3. **Sequential:** Activación en cascada (onda viajera)"""))

# ================ CELDA 11: Ejercicio 2 código ================
cells.append(nbf.v4.new_code_cell("""# EJERCICIO 2: Generar Spike Trains

def generate_spike_trains(n_neurons=20, duration=1000, base_rate=5.0,
                         correlation=0.3, pattern_type='random'):
    \"\"\"
    Genera spike trains sintéticos con diferentes patrones.

    Arguments:
    n_neurons -- número de neuronas
    duration -- duración en ms
    base_rate -- tasa de disparo base (Hz)
    correlation -- nivel de correlación (no usado en esta versión)
    pattern_type -- 'random', 'synchronized', 'sequential'

    Returns:
    spike_trains -- array (n_neurons, duration) con spikes
    \"\"\"
    spike_trains = np.zeros((n_neurons, duration))

    if pattern_type == 'random':
        # Actividad aleatoria independiente (Poisson)
        # Para cada neurona, generar spikes con tasa base_rate
        # (approx. 2 lines)
        # YOUR CODE STARTS HERE


        # YOUR CODE ENDS HERE

    elif pattern_type == 'synchronized':
        # Actividad sincronizada (patrón común)
        # Generar un patrón común y aplicarlo a todas (con 80% probabilidad)
        # (approx. 3 lines)
        # YOUR CODE STARTS HERE



        # YOUR CODE ENDS HERE

    elif pattern_type == 'sequential':
        # Actividad secuencial (onda de activación)
        # En cada tiempo t, activar neurona (t // 20) % n_neurons
        # (approx. 3 lines)
        # YOUR CODE STARTS HERE



        # YOUR CODE ENDS HERE

    return spike_trains"""))

# ================ CELDA 12: Test ejercicio 2 ================
cells.append(nbf.v4.new_code_cell("""# Test del Ejercicio 2
print("🔥 Generando spike trains con diferentes patrones...\\n")

patterns = ['random', 'synchronized', 'sequential']
spike_data = {}

for pattern in patterns:
    spikes = generate_spike_trains(n_neurons=15, duration=1000,
                                  pattern_type=pattern, base_rate=8.0)
    spike_data[pattern] = spikes
    print(f"   {pattern}: {spikes.shape}, total spikes={np.sum(spikes):.0f}")

# Test automático
test_generate_spike_trains(generate_spike_trains)"""))

# ================ CELDA 13: Ejercicio 3 intro ================
cells.append(nbf.v4.new_markdown_cell("""<a name='ex-3'></a>
### Ejercicio 3 - spike_trains_to_state_space

Convierte spike trains a representación en espacio de estados usando ventanas deslizantes.

**Instrucciones:**
1. Dividir el tiempo en ventanas de tamaño `bin_size`
2. Usar ventana deslizante con paso `stride`
3. Para cada ventana, contar spikes por neurona
4. Retornar matriz (n_bins x n_neurons)"""))

# ================ CELDA 14: Ejercicio 3 código ================
cells.append(nbf.v4.new_code_cell("""# EJERCICIO 3: Convertir Spike Trains a Espacio de Estados

def spike_trains_to_state_space(spike_trains, bin_size=50, stride=25):
    \"\"\"
    Convierte spike trains a representación en espacio de estados.

    Arguments:
    spike_trains -- matriz de spikes (n_neurons x time)
    bin_size -- tamaño de ventana en ms
    stride -- paso de la ventana deslizante

    Returns:
    state_space -- array (n_bins, n_neurons) con conteos de spikes
    \"\"\"
    n_neurons, duration = spike_trains.shape
    n_bins = (duration - bin_size) // stride + 1

    state_space = np.zeros((n_bins, n_neurons))

    # Para cada ventana, contar spikes por neurona
    # (approx. 4 lines)
    # YOUR CODE STARTS HERE




    # YOUR CODE ENDS HERE

    return state_space"""))

# ================ CELDA 15: Test ejercicio 3 ================
cells.append(nbf.v4.new_code_cell("""# Test del Ejercicio 3
print("🔄 Convirtiendo spike trains a espacio de estados...\\n")

state_spaces = {}
for pattern in patterns:
    state_spaces[pattern] = spike_trains_to_state_space(spike_data[pattern],
                                                         bin_size=50, stride=25)
    print(f"   {pattern}: {state_spaces[pattern].shape}")

# Test automático
test_spike_trains_to_state_space(spike_trains_to_state_space)"""))

# ================ CELDA 16: Visualización spike trains ================
cells.append(nbf.v4.new_code_cell("""# Visualizar spike trains y trayectorias
fig, axes = plt.subplots(3, 2, figsize=(18, 12))

for idx, pattern in enumerate(patterns):
    # Raster plot
    ax1 = axes[idx, 0]
    for neuron in range(spike_data[pattern].shape[0]):
        spike_times = np.where(spike_data[pattern][neuron] > 0)[0]
        ax1.scatter(spike_times, [neuron]*len(spike_times),
                   c='black', s=1, marker='|')
    ax1.set_ylabel('Neurona #', fontsize=11)
    ax1.set_xlabel('Tiempo (ms)', fontsize=11)
    ax1.set_title(f'Raster: {pattern.capitalize()}', fontsize=12, fontweight='bold')
    ax1.set_xlim([0, 1000])

    # Trayectoria 2D (PCA)
    ax2 = axes[idx, 1]
    if state_spaces[pattern].shape[0] > 2:
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(state_spaces[pattern])
        ax2.plot(reduced[:, 0], reduced[:, 1], 'o-', alpha=0.6, linewidth=2, markersize=4)
        ax2.scatter(reduced[0, 0], reduced[0, 1], c='green', s=200, marker='o',
                   zorder=5, edgecolors='black', linewidth=2, label='Inicio')
        ax2.scatter(reduced[-1, 0], reduced[-1, 1], c='red', s=200, marker='s',
                   zorder=5, edgecolors='black', linewidth=2, label='Final')
        ax2.set_title(f'Trayectoria: {pattern.capitalize()}', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()"""))

# ================ CELDA 17: Análisis topológico spikes ================
cells.append(nbf.v4.new_code_cell("""# Análisis topológico de spike trains
print("🔍 Analizando topología de patrones de disparo...\\n")

spike_diagrams = {}

for pattern in patterns:
    result = ripser(state_spaces[pattern], maxdim=1, thresh=15.0)
    spike_diagrams[pattern] = result['dgms']
    print(f"   {pattern}: H₁ = {len(result['dgms'][1])} ciclos")

# Visualizar
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, pattern in enumerate(patterns):
    plot_diagrams(spike_diagrams[pattern], ax=axes[idx])
    axes[idx].set_title(f'{pattern.capitalize()}\\nH₁ = {len(spike_diagrams[pattern][1])} ciclos',
                       fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()

print("\\n💡 El patrón SECUENCIAL muestra ciclos robustos (H₁ alto)")
print("   El patrón SINCRONIZADO tiene estructura simple (H₁ bajo)")"""))

# ================ CELDA 18: Caja resumen spike trains ================
cells.append(nbf.v4.new_markdown_cell("""<div style="background-color:#e3f2fd; padding:15px; border-left:5px solid #2196f3; margin: 20px 0;">

**💡 Lo que debes recordar:**

- **Spike trains** pueden analizarse mediante conversión a espacio de estados
- **Ventanas deslizantes** crean trayectoria en espacio neuronal
- **Patrones secuenciales** tienen estructura **cíclica** (H₁ > 0)
- **Patrones sincronizados** colapsan a **baja dimensión** (H₁ ≈ 0)
- TDA **detecta automáticamente** patrones temporales en actividad neuronal

</div>

---"""))

# ================ CELDA 19: Características para ML ================
cells.append(nbf.v4.new_markdown_cell("""<a name='6'></a>
## 6 - Características Topológicas para Machine Learning

[Volver al índice](#toc)

### 6.1 Vectorización de Diagramas

Para usar TDA en ML, convertimos diagramas a vectores de características:

1. **Número de características:** Contar puntos en el diagrama
2. **Persistencia máxima:** $\\max(death - birth)$
3. **Persistencia promedio:** $\\text{mean}(death - birth)$
4. **Entropía de persistencia:** $-\\sum p_i \\log(p_i)$ donde $p_i = \\frac{L_i}{\\sum L_j}$

---

<a name='ex-4'></a>
### Ejercicio 4 - extract_topological_features

Implementa la extracción de características escalares de un diagrama de persistencia.

**Instrucciones:**
1. Filtrar puntos infinitos del diagrama
2. Calcular lifetimes (persistencias)
3. Extraer estadísticas: n_features, max, mean, std, total
4. Calcular entropía de persistencia"""))

# ================ CELDA 20: Ejercicio 4 código ================
cells.append(nbf.v4.new_code_cell("""# EJERCICIO 4: Extraer Características Topológicas

def extract_topological_features(diagram, dim=1):
    \"\"\"
    Extrae características escalares de un diagrama de persistencia.

    Arguments:
    diagram -- lista de arrays [dgm_0, dgm_1, dgm_2, ...]
    dim -- dimensión homológica a analizar

    Returns:
    features -- diccionario con características
    \"\"\"
    features = {}

    # Verificar que el diagrama tenga datos
    if len(diagram[dim]) == 0:
        return {'n_features': 0, 'max_persistence': 0,
                'mean_persistence': 0, 'std_persistence': 0,
                'total_persistence': 0, 'entropy': 0}

    # Filtrar puntos infinitos
    # (approx. 1 line)
    # YOUR CODE STARTS HERE

    # YOUR CODE ENDS HERE

    if len(dgm) == 0:
        return {'n_features': 0, 'max_persistence': 0,
                'mean_persistence': 0, 'std_persistence': 0,
                'total_persistence': 0, 'entropy': 0}

    # Calcular lifetimes (persistencias)
    # lifetime = death - birth
    # (approx. 1 line)
    # YOUR CODE STARTS HERE

    # YOUR CODE ENDS HERE

    # Características básicas
    # (approx. 5 lines)
    # YOUR CODE STARTS HERE





    # YOUR CODE ENDS HERE

    # Entropía de persistencia
    # (approx. 4 lines)
    # YOUR CODE STARTS HERE




    # YOUR CODE ENDS HERE

    return features"""))

# ================ CELDA 21: Test ejercicio 4 ================
cells.append(nbf.v4.new_code_cell("""# Test del Ejercicio 4
print("📊 Extrayendo características topológicas...\\n")

feature_summary = []

for pattern in patterns:
    feats = extract_topological_features(spike_diagrams[pattern], dim=1)
    feats['pattern'] = pattern
    feature_summary.append(feats)

# Crear DataFrame
df_features = pd.DataFrame(feature_summary)
print("Características Topológicas (H₁):\\n")
print(df_features[['pattern', 'n_features', 'max_persistence', 'mean_persistence', 'entropy']])
print()

# Test automático
test_extract_topological_features_tutorial2(extract_topological_features)"""))

# ================ CELDA 22: Visualización características ================
cells.append(nbf.v4.new_code_cell("""# Visualizar características
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

features_to_plot = ['n_features', 'max_persistence', 'mean_persistence', 'entropy']
titles = ['Número de Ciclos', 'Máxima Persistencia',
          'Persistencia Promedio', 'Entropía']
colors = ['#e74c3c', '#3498db', '#2ecc71']

for idx, (feat, title) in enumerate(zip(features_to_plot, titles)):
    ax = axes[idx // 2, idx % 2]
    values = [df_features[df_features['pattern'] == p][feat].values[0] for p in patterns]
    bars = ax.bar(patterns, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_ylabel(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Patrón', fontsize=11)
    ax.set_title(f'{title} por Patrón', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()

print("✅ Cada patrón tiene una 'firma topológica' única!")"""))

# ================ CELDA 23: Caja resumen final ================
cells.append(nbf.v4.new_markdown_cell("""<div style="background-color:#e3f2fd; padding:15px; border-left:5px solid #2196f3; margin: 20px 0;">

**💡 Lo que debes recordar:**

- **Vectorización** permite usar TDA en pipelines de ML
- **Entropía de persistencia** mide la distribución de características
- **Cada patrón** tiene una **firma topológica** distintiva
- Características topológicas pueden usarse para:
  - Clasificar estados cerebrales
  - Detectar patrones anormales
  - Predecir transiciones de estado

</div>

---"""))

# ================ CELDA 24: Optimización ================
cells.append(nbf.v4.new_markdown_cell("""<a name='7'></a>
## 7 - Optimización y Resumen

[Volver al índice](#toc)

### 7.1 Estrategias de Optimización

Para datasets grandes (N > 500):

1. **Subsampling:** Reducir número de puntos
2. **PCA/UMAP:** Reducir dimensionalidad antes de TDA
3. **Threshold:** Limitar radio máximo
4. **Sparse matrices:** Solo distancias < threshold

### 7.2 Reglas prácticas:

- **N < 200:** Usa Ripser directamente
- **N > 200:** Considera PCA (5-10 dims) o subsampling
- **Dimensión alta (>20):** Siempre reduce con PCA primero
- **Threshold:** Usa 2-3× la escala típica de tus datos

---"""))

# ================ CELDA 25: Resumen ================
cells.append(nbf.v4.new_markdown_cell("""## 📝 Resumen

### ✅ Lo que dominamos:

1. **Filtraciones:** Vietoris-Rips es práctica para datos neuronales
2. **Distancias:** Bottleneck (robusta) vs Wasserstein (global)
3. **Spike trains:** Conversión a espacio de estados + TDA
4. **Características:** Vectorización para machine learning
5. **Optimización:** PCA, subsampling, threshold

### 🔑 Mensajes Clave:

- **TDA captura patrones temporales** en actividad neuronal
- **Ciclos (H₁)** revelan retroalimentación y patrones repetitivos
- **Firmas topológicas** son distintivas para cada tipo de actividad
- **Características topológicas** son robustas al ruido
- **Optimización** es crucial para aplicaciones reales

### 🧠 Impacto en Neurociencias:

TDA proporciona:
- Descripción **invariante** de patrones neuronales
- Detección **robusta** de estructuras funcionales
- Comparación **cuantitativa** entre condiciones
- Base para **biomarcadores** topológicos

---

## 🎉 ¡Felicitaciones!

Has completado el Tutorial 2. Ahora dominas técnicas avanzadas de homología persistente.

**¿Listo para el siguiente desafío?** → Tutorial 3: Conectividad Cerebral

---

**Autor:** MARK-126
**Última actualización:** 2025-01-15
**Licencia:** MIT"""))

# Agregar todas las celdas al notebook
nb['cells'] = cells

# Guardar notebook
with open('/home/user/TOPLOGIA-DATA-SCIENCE/notebooks/02_Homologia_Persistente_Avanzada_v2.ipynb', 'w') as f:
    nbf.write(nb, f)

print("✅ Tutorial 2 v2 creado exitosamente!")
