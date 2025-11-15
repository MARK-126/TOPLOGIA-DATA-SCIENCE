# Soluciones Tutorial 2: Homología Persistente Avanzada

Este documento contiene las soluciones completas y explicaciones detalladas para todos los ejercicios del Tutorial 2 v2.

---

## Ejercicio 1: generate_brain_state_realistic

### Solución Completa:

```python
def generate_brain_state_realistic(state_type, n_neurons=100, noise=0.1):
    """
    Genera estados cerebrales sintéticos con propiedades realistas.
    """
    if state_type == 'sleep':
        # Sueño: activación sincronizada, baja dimensionalidad
        base = np.random.randn(n_neurons, 1) @ np.random.randn(1, 5)
        data = base + np.random.randn(n_neurons, 5) * noise

    elif state_type == 'wakeful':
        # Vigilia: activación dispersa, alta dimensionalidad
        data = np.random.randn(n_neurons, 5) * 1.5

    elif state_type == 'attention':
        # Atención: subredes focales activas
        data = np.zeros((n_neurons, 5))
        data[:n_neurons//3] = np.random.randn(n_neurons//3, 5) * 2.0
        data[n_neurons//3:] = np.random.randn(2*n_neurons//3, 5) * 0.3

    elif state_type == 'memory':
        # Memoria: patrones cíclicos (bucles de retroalimentación)
        theta = np.linspace(0, 4*np.pi, n_neurons)
        data = np.column_stack([
            np.cos(theta),
            np.sin(theta),
            np.cos(2*theta) * 0.5,
            np.sin(2*theta) * 0.5,
            np.random.randn(n_neurons) * noise
        ])

    return data
```

### Explicación Paso a Paso:

#### **Sleep (Sueño):**
- **Objetivo:** Modelar activación sincronizada de baja dimensionalidad
- **Método:** Proyección desde 1D a 5D
  1. `np.random.randn(n_neurons, 1)` crea un vector columna (factor común)
  2. `@ np.random.randn(1, 5)` proyecta a 5 dimensiones
  3. El resultado: todas las neuronas varían juntas (correlacionadas)
  4. `+ noise` agrega variación individual pequeña

**Intuición neurobiológica:** Durante el sueño, las neuronas tienden a activarse de forma sincronizada (ondas lentas), reduciendo la dimensionalidad efectiva del espacio de estados.

#### **Wakeful (Vigilia):**
- **Objetivo:** Activación dispersa e independiente
- **Método:** Simple ruido gaussiano
  - `np.random.randn(n_neurons, 5) * 1.5` genera puntos aleatorios en 5D
  - Cada neurona es independiente
  - Factor 1.5 aumenta la dispersión

**Intuición neurobiológica:** Durante vigilia activa, las neuronas tienen patrones de activación más diversos y menos correlacionados.

#### **Attention (Atención):**
- **Objetivo:** Subred focal altamente activa
- **Método:** Activación diferencial
  1. `data = np.zeros((n_neurons, 5))` inicializa todo en 0
  2. Primera tercera parte: alta actividad (`* 2.0`)
  3. Resto: actividad basal baja (`* 0.3`)

**Intuición neurobiológica:** La atención selectiva implica que una subred específica (ej., corteza prefrontal) está muy activa mientras otras regiones tienen actividad basal.

#### **Memory (Memoria):**
- **Objetivo:** Estructura cíclica (bucles de retroalimentación)
- **Método:** Funciones periódicas
  1. `theta = np.linspace(0, 4*np.pi, n_neurons)` crea ángulos
  2. `cos(theta), sin(theta)` forman un bucle principal
  3. `cos(2*theta), sin(2*theta)` agregan un segundo armónico
  4. Quinta dimensión: ruido

**Intuición neurobiológica:** Los bucles de retroalimentación en redes neuronales (como en memoria de trabajo) crean trayectorias cíclicas en el espacio de estados.

### Consejos de Debugging:

**Error común 1:** `ValueError: operands could not be broadcast`
- **Causa:** Dimensiones incompatibles en multiplicación de matrices
- **Solución:** Verifica que `(n_neurons, 1) @ (1, 5)` produce `(n_neurons, 5)`

**Error común 2:** `TypeError: unsupported operand type(s) for @`
- **Causa:** Usando `*` en lugar de `@` para multiplicación matricial
- **Solución:** Usa `@` para multiplicación de matrices, no `*`

**Error común 3:** Estado 'memory' no tiene ciclos
- **Causa:** `theta` no tiene rango suficiente para completar ciclos
- **Solución:** Usa `4*np.pi` o más para hacer al menos 2 ciclos completos

---

## Ejercicio 2: generate_spike_trains

### Solución Completa:

```python
def generate_spike_trains(n_neurons=20, duration=1000, base_rate=5.0,
                         correlation=0.3, pattern_type='random'):
    """
    Genera spike trains sintéticos con diferentes patrones.
    """
    spike_trains = np.zeros((n_neurons, duration))

    if pattern_type == 'random':
        # Actividad aleatoria independiente
        for i in range(n_neurons):
            spike_trains[i] = poisson.rvs(base_rate/1000, size=duration)

    elif pattern_type == 'synchronized':
        # Actividad sincronizada
        common_pattern = poisson.rvs(base_rate/1000, size=duration)
        for i in range(n_neurons):
            spike_trains[i] = common_pattern * (np.random.rand(duration) < 0.8)

    elif pattern_type == 'sequential':
        # Actividad secuencial
        for t in range(duration):
            active_neuron = (t // 20) % n_neurons
            spike_trains[active_neuron, t] = poisson.rvs(base_rate*3/1000)

    return spike_trains
```

### Explicación Paso a Paso:

#### **Random (Aleatorio):**
- **Distribución de Poisson:** Modelo estándar para spikes neuronales
- `base_rate/1000` convierte Hz a probabilidad por ms
- `size=duration` genera un spike train completo
- Cada neurona es **independiente**

**Matemática:**
$$P(\\text{spike en t}) = \\lambda \\Delta t$$
donde $\\lambda$ = `base_rate` (Hz), $\\Delta t$ = 1 ms

#### **Synchronized (Sincronizado):**
- `common_pattern`: Patrón maestro compartido por todas
- `* (np.random.rand(duration) < 0.8)`: Cada neurona sigue el patrón con 80% de probabilidad
- Resultado: Alta correlación entre neuronas

**¿Por qué 80%?** Para simular sincronización imperfecta (realismo biológico). Sincronización perfecta (100%) es rara en el cerebro real.

#### **Sequential (Secuencial):**
- `(t // 20) % n_neurons`: Elige qué neurona activa en cada tiempo
  - `t // 20`: Cambia cada 20 ms
  - `% n_neurons`: Cicla entre todas las neuronas
- `base_rate*3`: La neurona activa dispara más frecuentemente
- Resultado: Onda de activación que recorre la población

**Visualización:**
```
t=0-19:   Neurona 0 activa  ███░░░░░░░
t=20-39:  Neurona 1 activa  ░░░███░░░░░
t=40-59:  Neurona 2 activa  ░░░░░░███░░
...y así sucesivamente
```

### Consejos de Debugging:

**Error común 1:** `AttributeError: module 'scipy.stats' has no attribute 'poisson'`
- **Causa:** No importaste `from scipy.stats import poisson`
- **Solución:** Asegúrate de tener la importación correcta

**Error común 2:** Spikes demasiado frecuentes (matriz llena de 1s)
- **Causa:** No dividiste `base_rate` por 1000
- **Solución:** `base_rate/1000` para convertir Hz → probabilidad/ms

**Error común 3:** Patrón secuencial no es visible
- **Causa:** Cambio muy rápido (`t // 5`) o muy lento (`t // 100`)
- **Solución:** Usa `t // 20` para ~20 ms por neurona

---

## Ejercicio 3: spike_trains_to_state_space

### Solución Completa:

```python
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
```

### Explicación Paso a Paso:

#### **Cálculo del número de bins:**
```python
n_bins = (duration - bin_size) // stride + 1
```
- Ejemplo: `duration=1000`, `bin_size=50`, `stride=25`
- `(1000 - 50) // 25 + 1 = 950 // 25 + 1 = 38 + 1 = 39 bins`

**¿Por qué esta fórmula?**
- Primera ventana: `[0, 50)`
- Segunda ventana: `[25, 75)` (overlap de 25 ms)
- Última ventana: `[950, 1000)`
- Total: 39 ventanas

#### **Ventana deslizante:**
```python
for i in range(n_bins):
    start = i * stride          # 0, 25, 50, 75, ...
    end = start + bin_size      # 50, 75, 100, 125, ...
```

#### **Conteo de spikes:**
```python
state_space[i] = np.sum(spike_trains[:, start:end], axis=1)
```
- `spike_trains[:, start:end]`: Todas las neuronas, ventana temporal
- `axis=1`: Suma a lo largo del tiempo
- Resultado: Vector de conteos (un valor por neurona)

**Visualización:**
```
spike_trains:
Neurona 0: |-----|---------|---|-----|--------|
Neurona 1: |---|-----|----------|--|----------|
           [   Bin 1  ][  Bin 2   ][  Bin 3  ]

state_space:
Bin 1 → [3, 2]  # Neurona 0: 3 spikes, Neurona 1: 2 spikes
Bin 2 → [1, 3]
Bin 3 → [2, 2]
```

### Consejos de Debugging:

**Error común 1:** Número incorrecto de bins
- **Causa:** Fórmula incorrecta `duration // stride`
- **Solución:** Usa `(duration - bin_size) // stride + 1`

**Error común 2:** `IndexError: index out of bounds`
- **Causa:** Última ventana excede `duration`
- **Solución:** Asegúrate de que `end = start + bin_size` esté dentro de límites

**Error común 3:** Valores muy altos en `state_space`
- **Causa:** Sumaste sobre el eje incorrecto
- **Solución:** Usa `axis=1` para sumar a lo largo del tiempo, no neuronas

---

## Ejercicio 4: extract_topological_features

### Solución Completa:

```python
def extract_topological_features(diagram, dim=1):
    """
    Extrae características escalares de un diagrama de persistencia.
    """
    features = {}

    if len(diagram[dim]) == 0:
        return {'n_features': 0, 'max_persistence': 0,
                'mean_persistence': 0, 'std_persistence': 0,
                'total_persistence': 0, 'entropy': 0}

    # Filtrar puntos infinitos
    dgm = diagram[dim][np.isfinite(diagram[dim][:, 1])]

    if len(dgm) == 0:
        return {'n_features': 0, 'max_persistence': 0,
                'mean_persistence': 0, 'std_persistence': 0,
                'total_persistence': 0, 'entropy': 0}

    # Calcular lifetimes
    lifetimes = dgm[:, 1] - dgm[:, 0]

    # Características básicas
    features['n_features'] = len(dgm)
    features['max_persistence'] = np.max(lifetimes)
    features['mean_persistence'] = np.mean(lifetimes)
    features['std_persistence'] = np.std(lifetimes)
    features['total_persistence'] = np.sum(lifetimes)

    # Entropía de persistencia
    if np.sum(lifetimes) > 0:
        probs = lifetimes / np.sum(lifetimes)
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        features['entropy'] = entropy
    else:
        features['entropy'] = 0

    return features
```

### Explicación Paso a Paso:

#### **1. Manejo de diagramas vacíos:**
```python
if len(diagram[dim]) == 0:
    return {all zeros}
```
- **Crucial:** Sin esto, código falla con `IndexError`
- Retorna características nulas para indicar "sin topología"

#### **2. Filtrar puntos infinitos:**
```python
dgm = diagram[dim][np.isfinite(diagram[dim][:, 1])]
```
- **¿Por qué?** Puntos con `death = ∞` representan características que nunca mueren
- Estas son usualmente características H₀ que persisten siempre
- No podemos calcular `lifetime = ∞ - birth`
- `np.isfinite(...)` retorna `True` solo para valores finitos

#### **3. Calcular lifetimes (persistencias):**
```python
lifetimes = dgm[:, 1] - dgm[:, 0]
```
- `dgm[:, 0]`: Columna de births
- `dgm[:, 1]`: Columna de deaths
- **Interpretación:** ¿Cuánto "vivió" cada característica topológica?

**Ejemplo:**
```
Point: (birth=0.2, death=0.8) → lifetime = 0.6 (muy persistente)
Point: (birth=0.5, death=0.51) → lifetime = 0.01 (ruido)
```

#### **4. Estadísticas básicas:**
- `max_persistence`: Característica más robusta
- `mean_persistence`: Persistencia típica
- `std_persistence`: Variabilidad
- `total_persistence`: "Cantidad total" de topología

#### **5. Entropía de persistencia:**
```python
probs = lifetimes / np.sum(lifetimes)
entropy = -np.sum(probs * np.log(probs + 1e-10))
```

**Interpretación:**
- Normaliza lifetimes a distribución de probabilidad
- Calcula entropía de Shannon: $H = -\\sum p_i \\log(p_i)$
- **Entropía alta:** Muchas características de persistencia similar
- **Entropía baja:** Una o pocas características dominantes

**Ejemplo:**
```
Caso 1: lifetimes = [0.1, 0.1, 0.1, 0.1, 0.1]
→ probs = [0.2, 0.2, 0.2, 0.2, 0.2]
→ entropy ≈ 1.6 (ALTA - uniforme)

Caso 2: lifetimes = [0.9, 0.01, 0.01, 0.01, 0.07]
→ probs ≈ [0.9, 0.01, 0.01, 0.01, 0.07]
→ entropy ≈ 0.5 (BAJA - dominada por una)
```

**¿Por qué +1e-10?** Evitar `log(0) = -∞` en caso de probabilidades exactamente 0.

### Consejos de Debugging:

**Error común 1:** `RuntimeWarning: divide by zero in log`
- **Causa:** No agregaste `+ 1e-10` en el logaritmo
- **Solución:** Siempre usa `np.log(probs + 1e-10)`

**Error común 2:** `IndexError: index 1 is out of bounds`
- **Causa:** Accediste `diagram[dim]` pero `dim` está fuera de rango
- **Solución:** Verifica que el diagrama tenga al menos `dim+1` dimensiones

**Error común 3:** Entropía negativa
- **Causa:** Error en la fórmula (signo)
- **Solución:** Debe ser `-np.sum(...)` (negativo)

---

## 🎓 Ejercicios Adicionales (Desafíos)

### Desafío 1: Distancias entre Estados
Calcula la matriz de distancias Bottleneck entre todos los pares de estados cerebrales.

**Pista:**
```python
from persim import bottleneck

for i in range(len(states)):
    for j in range(i+1, len(states)):
        dist = bottleneck(diagrams[i][1], diagrams[j][1])
```

### Desafío 2: Clasificador Topológico
Usa características topológicas para entrenar un Random Forest que clasifique patrones de spike trains.

**Pista:**
```python
from sklearn.ensemble import RandomForestClassifier

# Generar múltiples ejemplos
X = []  # características topológicas
y = []  # etiquetas de patrón

# Entrenar
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
```

### Desafío 3: Evolución Temporal
Estudia cómo cambia la topología durante una transición de estados.

**Pista:**
```python
# Crear transición gradual
alpha = np.linspace(0, 1, 50)
for a in alpha:
    mixed = (1-a)*state1 + a*state2
    # Calcular topología
```

---

## 📚 Recursos Adicionales

### Papers Recomendados:
1. Giusti et al. (2015). "Clique topology reveals intrinsic structure in neural correlations"
2. Petri et al. (2014). "Homological scaffolds of brain functional networks"
3. Sizemore et al. (2019). "Cliques and cavities in the human connectome"

### Documentación:
- [Ripser](https://ripser.scikit-tda.org/): Homología persistente rápida
- [Persim](https://persim.scikit-tda.org/): Distancias entre diagramas
- [Giotto-TDA](https://giotto-ai.github.io/gtda-docs/): Suite completa

---

## 🤝 ¿Preguntas?

Si tienes dudas sobre las soluciones:
1. Revisa los comentarios en el código
2. Compara con los tests automáticos
3. Consulta la documentación de las bibliotecas
4. Abre un issue en el repositorio

**¡Buen trabajo completando el Tutorial 2!** 🎉

---

**Última actualización:** 2025-01-15
**Autor:** MARK-126
