# Soluciones: Tutorial 1 - Introducción al TDA

Este archivo contiene las soluciones completas para los ejercicios del Tutorial 1 versión interactiva.

**⚠️ IMPORTANTE:** Intenta resolver los ejercicios por tu cuenta antes de consultar estas soluciones.

---

## Ejercicio 1 - build_simplicial_complex

### Solución:

```python
def build_simplicial_complex(points, epsilon):
    n_points = len(points)
    distances = squareform(pdist(points))
    edges = []

    # Paso 1: Conectar puntos cercanos
    for i in range(n_points):
        for j in range(i+1, n_points):
            if distances[i, j] <= epsilon:
                edges.append((i, j))

    # Paso 2: Encontrar triángulos
    triangles = []
    for i in range(n_points):
        for j in range(i+1, n_points):
            for k in range(j+1, n_points):
                if (distances[i,j] <= epsilon and
                    distances[j,k] <= epsilon and
                    distances[i,k] <= epsilon):
                    triangles.append([i, j, k])

    return edges, triangles
```

### Explicación:

1. **Paso 1 - Aristas:**
   - Iteramos sobre todos los pares de puntos (i, j) donde i < j
   - Si la distancia euclidiana es ≤ epsilon, agregamos la arista

2. **Paso 2 - Triángulos:**
   - Iteramos sobre todas las ternas (i, j, k)
   - Verificamos que TODAS las distancias por pares sean ≤ epsilon
   - Si los tres puntos están conectados, formamos un triángulo (2-simplejo)

---

## Ejercicio 2 - compute_betti_numbers

### Solución:

```python
def compute_betti_numbers(points, max_epsilon=2.0, num_steps=50):
    epsilons = np.linspace(0.01, max_epsilon, num_steps)
    betti_0 = np.zeros(num_steps)
    betti_1 = np.zeros(num_steps)
    betti_2 = np.zeros(num_steps)

    # Calcular homología persistente una sola vez
    result = ripser(points, maxdim=2, thresh=max_epsilon)
    diagrams = result['dgms']

    # Para cada epsilon, contar características que existen
    for i, eps in enumerate(epsilons):
        # Dimensión 0 (componentes)
        betti_0[i] = np.sum((diagrams[0][:, 0] <= eps) &
                           ((diagrams[0][:, 1] > eps) | np.isinf(diagrams[0][:, 1])))

        # Dimensión 1 (ciclos)
        if len(diagrams) > 1:
            betti_1[i] = np.sum((diagrams[1][:, 0] <= eps) &
                               (diagrams[1][:, 1] > eps))

        # Dimensión 2 (cavidades)
        if len(diagrams) > 2:
            betti_2[i] = np.sum((diagrams[2][:, 0] <= eps) &
                               (diagrams[2][:, 1] > eps))

    return epsilons, betti_0, betti_1, betti_2
```

### Explicación:

1. **Homología Persistente:**
   - Calculamos la homología UNA SOLA VEZ con ripser
   - Esto nos da los "tiempos de nacimiento y muerte" de todas las características

2. **Conteo de Features:**
   - Para cada epsilon, contamos features que:
     - Nacieron antes o en epsilon (birth <= eps)
     - Y aún están vivas (death > eps) o son infinitas

3. **Dimensiones:**
   - diagrams[0]: H₀ (componentes)
   - diagrams[1]: H₁ (ciclos)
   - diagrams[2]: H₂ (cavidades)

---

## Ejercicio 3 - generate_neural_network

### Solución:

```python
def generate_neural_network(n_neurons=50, connectivity=0.3, noise_level=0.1):
    # Crear dos comunidades de neuronas
    community1 = np.random.randn(n_neurons//2, 2) * 0.5 + np.array([0, 0])
    community2 = np.random.randn(n_neurons//2, 2) * 0.5 + np.array([3, 0])

    # Agregar una neurona puente
    bridge = np.array([[1.5, 0]])

    # Combinar
    neurons = np.vstack([community1, community2, bridge])

    # Agregar ruido
    neurons += np.random.randn(*neurons.shape) * noise_level

    return neurons
```

### Explicación:

1. **Comunidad 1:**
   - Centrada en [0, 0]
   - `np.random.randn` genera puntos gaussianos
   - Escalamos por 0.5 para concentrarlos

2. **Comunidad 2:**
   - Centrada en [3, 0]
   - Separada ~3 unidades de la primera

3. **Neurona Puente:**
   - Posicionada en [1.5, 0] (punto medio)
   - Conecta ambas comunidades

4. **Ruido:**
   - Agrega variabilidad realista
   - `noise_level` controla la intensidad

---

## Ejercicio 4 - generate_brain_state

### Solución:

```python
def generate_brain_state(state_type='resting', n_neurons=100):
    if state_type == 'resting':
        # Estado de reposo: activación dispersa
        data = np.random.randn(n_neurons, 3) * 1.5

    elif state_type == 'active':
        # Estado activo: estructura más organizada (esfera)
        theta = np.random.uniform(0, 2*np.pi, n_neurons)
        phi = np.random.uniform(0, np.pi, n_neurons)
        r = 1 + np.random.randn(n_neurons) * 0.1

        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)

        data = np.column_stack([x, y, z])

    return data
```

### Explicación:

**Estado Resting:**
- Datos gaussianos 3D sin estructura
- Simula activación neuronal aleatoria en reposo

**Estado Active (Esfera):**
1. **Coordenadas Esféricas:**
   - θ (theta): ángulo azimutal [0, 2π]
   - φ (phi): ángulo polar [0, π]
   - r: radio ~1 con pequeño ruido

2. **Conversión a Cartesianas:**
   - x = r × sin(φ) × cos(θ)
   - y = r × sin(φ) × sin(θ)
   - z = r × cos(φ)

3. **Resultado:**
   - Puntos distribuidos uniformemente en superficie de esfera
   - Simula activación neuronal organizada/estructurada

---

## 🎯 Consejos para Debugging

### Si tu código no pasa los tests:

1. **Revisa índices:**
   - Python usa 0-indexing
   - `range(i+1, n)` excluye i para evitar duplicados

2. **Verifica condiciones:**
   - `<=` vs `<` puede cambiar resultados
   - Maneja casos especiales (diagramas vacíos)

3. **Print debugging:**
   ```python
   print(f"Distancia {i}-{j}: {distances[i,j]}")
   print(f"Epsilon: {epsilon}")
   ```

4. **Usa tests unitarios:**
   - Los tests te dicen exactamente qué falla
   - Lee los mensajes de error cuidadosamente

---

## 📚 Conceptos Clave para Recordar

1. **Complejo Simplicial:**
   - Construcción bottom-up: puntos → aristas → triángulos
   - Parámetro ε controla densidad

2. **Homología Persistente:**
   - Rastrea características a través de múltiples escalas
   - Birth/death times indican importancia

3. **Números de Betti:**
   - β₀: componentes (siempre comienza alto, converge a 1)
   - β₁: ciclos (detecta loops/circuitos)
   - β₂: cavidades (detecta estructuras volumétricas)

4. **Aplicación Neural:**
   - Neuronas → vértices
   - Conexiones funcionales → aristas
   - Circuitos recurrentes → ciclos (β₁)

---

## 🚀 Ejercicios Adicionales (Opcional)

Si terminaste rápido, intenta estos desafíos:

### Desafío 1: Optimizar `build_simplicial_complex`
```python
# Hint: Usa vectorización de NumPy en lugar de loops
# Busca: np.where, broadcasting
```

### Desafío 2: Visualizar Evolución de Complejo
```python
# Crea una animación mostrando cómo crece el complejo
# al aumentar epsilon de 0.1 a 2.0
```

### Desafío 3: Aplicar a Datos Reales
```python
# Descarga datos de conectividad cerebral real
# Aplica TDA y compara con datos sintéticos
```

---

**¿Preguntas?** Consulta:
- [Documentación Ripser](https://ripser.scikit-tda.org/)
- [Tutorial de homología](https://www.math3ma.com/blog/what-is-homology)
- O abre un issue en el repositorio

**¡Sigue adelante con el Tutorial 2!** 🎓
