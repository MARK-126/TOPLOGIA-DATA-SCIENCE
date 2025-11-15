#!/usr/bin/env python3
"""
Script para crear Tutorial 5 v2: Series Temporales EEG (Versión Interactiva)
"""

import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = []

# ========== HEADER ==========
cells.append(nbf.v4.new_markdown_cell("""# Tutorial 5: Series Temporales con TDA (Versión Interactiva)

## Análisis Topológico de Señales EEG Temporales

**Autor:** MARK-126
**Nivel:** Avanzado
**Tiempo estimado:** 150-180 minutos

---

## 🎯 Objetivos de Aprendizaje

1. ✅ Dominar embeddings de Takens
2. ✅ Aplicar TDA a señales EEG
3. ✅ Detectar eventos con ventanas deslizantes
4. ✅ Clasificar estados cognitivos
5. ✅ Extraer biomarcadores temporales

---

## ⚠️ Nota sobre Ejercicios

Este notebook contiene **3 ejercicios interactivos** sobre análisis temporal.

---"""))

# ========== TABLE OF CONTENTS ==========
cells.append(nbf.v4.new_markdown_cell("""<a name='toc'></a>
## 📚 Tabla de Contenidos

- [1 - Setup e Importaciones](#1)
- [2 - Teorema de Takens](#2)
    - [Ejercicio 1 - takens_embedding](#ex-1)
- [3 - Generación de Señales EEG](#3)
- [4 - Ventanas Deslizantes](#4)
    - [Ejercicio 2 - sliding_window_persistence](#ex-2)
- [5 - Clasificación de Estados](#5)
    - [Ejercicio 3 - classify_states_with_tda](#ex-3)
- [6 - Resumen](#6)

---"""))

# ========== SECTION 1: SETUP ==========
cells.append(nbf.v4.new_markdown_cell("""<a name='1'></a>
## 1 - Setup e Importaciones

[Volver al índice](#toc)"""))

cells.append(nbf.v4.new_code_cell("""import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from ripser import ripser
from persim import plot_diagrams
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.stats import zscore
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd

from tda_tests import (
    test_takens_embedding,
    test_sliding_window_persistence,
    test_classify_states_with_tda
)

plt.style.use('seaborn-v0_8-whitegrid')
np.random.seed(42)
print("✅ Setup completado")"""))

# ========== SECTION 2: EXERCISE 1 ==========
cells.append(nbf.v4.new_markdown_cell("""<a name='2'></a>
## 2 - Teorema de Takens y Embeddings

[Volver al índice](#toc)

### El Problema

Tenemos una **serie temporal 1D**, pero TDA necesita datos multi-dimensionales.

### Solución: Teorema de Takens (1981)

Dada serie temporal $x(t)$, reconstruir espacio de estados:

$$\\mathbf{y}(t) = [x(t), x(t+\\tau), x(t+2\\tau), ..., x(t+(d-1)\\tau)]$$

Donde:
- **τ (tau):** Delay (retraso temporal)
- **d:** Dimensión de embedding

<a name='ex-1'></a>
### Ejercicio 1 - takens_embedding

Implementa embedding de Takens con estimación automática de delay.

**Pasos:**
1. Estimar delay óptimo usando autocorrelación
2. Crear matriz de embedding
3. Validar dimensionalidad"""))

cells.append(nbf.v4.new_code_cell("""# EJERCICIO 1: Embedding de Takens

def takens_embedding(timeseries, delay=None, dimension=3):
    \"\"\"
    Crea embedding de Takens de una serie temporal.

    Arguments:
    timeseries -- array 1D de la señal temporal
    delay -- retraso τ (None = estimar automáticamente)
    dimension -- dimensión del embedding

    Returns:
    embedded -- array (n_points, dimension)
    delay_used -- delay utilizado
    \"\"\"

    # 1. Estimar delay si no se proporciona
    # Calcular autocorrelación y encontrar primer mínimo
    # (approx. 8 lines)
    # YOUR CODE STARTS HERE








    # YOUR CODE ENDS HERE

    # 2. Crear matriz de embedding
    # Cada fila: [x(t), x(t+τ), x(t+2τ), ..., x(t+(d-1)τ)]
    # (approx. 7 lines)
    # YOUR CODE STARTS HERE







    # YOUR CODE ENDS HERE

    return embedded, delay_used"""))

cells.append(nbf.v4.new_code_cell("""# Test Ejercicio 1
# Generar señal de prueba (sistema de Lorenz)
def lorenz(xyz, t, sigma=10, rho=28, beta=8/3):
    x, y, z = xyz
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

from scipy.integrate import odeint
t = np.linspace(0, 50, 5000)
xyz = odeint(lorenz, [1.0, 1.0, 1.0], t)
x_signal = xyz[:, 0]

embedded, delay = takens_embedding(x_signal, dimension=3)
print(f"✅ Embedding: {embedded.shape}")
print(f"   Delay: τ={delay}")
test_takens_embedding(takens_embedding)"""))

# ========== SECTION 3: EEG GENERATION ==========
cells.append(nbf.v4.new_markdown_cell("""<a name='3'></a>
## 3 - Generación de Señales EEG

[Volver al índice](#toc)

Generamos señales EEG sintéticas con diferentes estados:
- **Normal:** Alpha (8-13 Hz) + Beta (13-30 Hz)
- **Seizure:** Spike-wave a 3 Hz
- **Sleep:** Delta (0.5-4 Hz)"""))

cells.append(nbf.v4.new_code_cell("""def generate_eeg_signal(duration=10, fs=250, state='normal'):
    \"\"\"Genera señal EEG sintética.\"\"\"
    n_samples = int(duration * fs)
    t = np.linspace(0, duration, n_samples)

    if state == 'normal':
        alpha = 0.5 * np.sin(2 * np.pi * 10 * t)
        beta = 0.3 * np.sin(2 * np.pi * 20 * t)
        noise = 0.2 * np.random.randn(n_samples)
        eeg = alpha + beta + noise

    elif state == 'seizure':
        spike_wave = 2.0 * np.sin(2 * np.pi * 3 * t)
        harmonics = 0.5 * np.sin(2 * np.pi * 6 * t)
        noise = 0.1 * np.random.randn(n_samples)
        eeg = spike_wave + harmonics + noise

    elif state == 'sleep':
        delta = 1.5 * np.sin(2 * np.pi * 2 * t)
        theta = 0.3 * np.sin(2 * np.pi * 6 * t)
        noise = 0.15 * np.random.randn(n_samples)
        eeg = delta + theta + noise

    return t, eeg

# Generar señales
t_normal, eeg_normal = generate_eeg_signal(duration=10, state='normal')
t_seizure, eeg_seizure = generate_eeg_signal(duration=10, state='seizure')
t_sleep, eeg_sleep = generate_eeg_signal(duration=10, state='sleep')

print(f"✅ Señales generadas: {len(eeg_normal)} muestras cada una")"""))

# ========== SECTION 4: EXERCISE 2 ==========
cells.append(nbf.v4.new_markdown_cell("""<a name='4'></a>
## 4 - Análisis con Ventanas Deslizantes

[Volver al índice](#toc)

<a name='ex-2'></a>
### Ejercicio 2 - sliding_window_persistence

Detecta eventos usando ventanas deslizantes y persistencia.

**Estrategia:**
1. Dividir señal en ventanas solapadas
2. Calcular embedding de Takens en cada ventana
3. Extraer persistencia H₁
4. Detectar cambios topológicos"""))

cells.append(nbf.v4.new_code_cell("""# EJERCICIO 2: Ventanas Deslizantes

def sliding_window_persistence(signal, window_size=500, stride=100, fs=250):
    \"\"\"
    Analiza señal con ventanas deslizantes y TDA.

    Arguments:
    signal -- señal temporal 1D
    window_size -- tamaño de ventana (muestras)
    stride -- paso de desplazamiento
    fs -- frecuencia de muestreo

    Returns:
    time_points -- puntos temporales centrales
    n_cycles_over_time -- número de ciclos H₁ en cada ventana
    max_persistence_over_time -- persistencia máxima en cada ventana
    \"\"\"
    n_samples = len(signal)
    time_points = []
    n_cycles_over_time = []
    max_persistence_over_time = []

    # 1. Iterar sobre ventanas deslizantes
    # (approx. 2 lines)
    # YOUR CODE STARTS HERE


    # YOUR CODE ENDS HERE

        # 2. Extraer ventana y normalizar
        # (approx. 3 lines)
        # YOUR CODE STARTS HERE



        # YOUR CODE ENDS HERE

        # 3. Crear embedding de Takens
        # (approx. 2 lines)
        # YOUR CODE STARTS HERE


        # YOUR CODE ENDS HERE

        # 4. Calcular persistencia H₁
        # (approx. 5 lines)
        # YOUR CODE STARTS HERE





        # YOUR CODE ENDS HERE

        # 5. Guardar resultados
        # (approx. 3 lines)
        # YOUR CODE STARTS HERE



        # YOUR CODE ENDS HERE

    return np.array(time_points), np.array(n_cycles_over_time), np.array(max_persistence_over_time)"""))

cells.append(nbf.v4.new_code_cell("""# Test Ejercicio 2
time_pts, n_cycles, max_pers = sliding_window_persistence(eeg_normal,
                                                          window_size=500,
                                                          stride=100)
print(f"✅ Ventanas analizadas: {len(time_pts)}")
print(f"   Ciclos promedio: {np.mean(n_cycles):.1f}")
test_sliding_window_persistence(sliding_window_persistence)"""))

# ========== SECTION 5: EXERCISE 3 ==========
cells.append(nbf.v4.new_markdown_cell("""<a name='5'></a>
## 5 - Clasificación de Estados Cerebrales

[Volver al índice](#toc)

<a name='ex-3'></a>
### Ejercicio 3 - classify_states_with_tda

Clasifica estados cerebrales usando características topológicas.

**Features a extraer:**
- TDA: n_cycles, max_persistence, mean_persistence
- Espectrales: potencia en bandas (delta, alpha, beta)
- Temporales: media, std"""))

cells.append(nbf.v4.new_code_cell("""# EJERCICIO 3: Clasificación

def classify_states_with_tda(signals_dict, test_size=0.3):
    \"\"\"
    Clasifica estados cerebrales usando TDA + features espectrales.

    Arguments:
    signals_dict -- diccionario {state_name: list_of_signals}
    test_size -- proporción de test

    Returns:
    clf -- clasificador entrenado
    accuracy -- precisión en test
    report -- reporte de clasificación
    \"\"\"
    X_data = []
    y_data = []
    state_names = list(signals_dict.keys())

    # 1. Extraer features de cada señal
    # (approx. 12 lines para embedding + persistencia + espectrales)
    # YOUR CODE STARTS HERE












    # YOUR CODE ENDS HERE

    # 2. Convertir a arrays
    # (approx. 2 lines)
    # YOUR CODE STARTS HERE


    # YOUR CODE ENDS HERE

    # 3. Dividir train/test
    # (approx. 2 lines)
    # YOUR CODE STARTS HERE


    # YOUR CODE ENDS HERE

    # 4. Entrenar Random Forest
    # (approx. 2 lines)
    # YOUR CODE STARTS HERE


    # YOUR CODE ENDS HERE

    # 5. Evaluar
    # (approx. 3 lines)
    # YOUR CODE STARTS HERE



    # YOUR CODE ENDS HERE

    return clf, accuracy, report"""))

cells.append(nbf.v4.new_code_cell("""# Generar dataset
signals_dataset = {
    'normal': [generate_eeg_signal(duration=5, state='normal')[1] for _ in range(50)],
    'seizure': [generate_eeg_signal(duration=5, state='seizure')[1] for _ in range(50)],
    'sleep': [generate_eeg_signal(duration=5, state='sleep')[1] for _ in range(50)]
}

# Test Ejercicio 3
clf, acc, report = classify_states_with_tda(signals_dataset)
print(f"✅ Clasificador entrenado")
print(f"   Precisión: {acc:.1%}")
print(f"\\n{report}")
test_classify_states_with_tda(classify_states_with_tda)"""))

# ========== SECTION 6: SUMMARY ==========
cells.append(nbf.v4.new_markdown_cell("""<a name='6'></a>
## 6 - Resumen

[Volver al índice](#toc)

<div style="background-color:#e3f2fd; padding:15px; border-left:5px solid #2196f3;">

**💡 Lo que aprendimos:**

- **Teorema de Takens** reconstruye dinámica desde señal 1D
- **Embeddings** transforman series temporales en nubes de puntos
- **Ventanas deslizantes** detectan eventos temporales
- **TDA + ML** clasifica estados cerebrales con alta precisión
- **Aplicaciones clínicas** incluyen epilepsia, sueño, anestesia

</div>

---

### 🧠 Aplicaciones Clínicas

**Epilepsia:** Detección temprana de crisis (cambios topológicos)
**Sueño:** Clasificación automática de etapas
**Anestesia:** Monitoreo de profundidad
**Coma:** Evaluación de nivel de consciencia

---

## 🎉 ¡Excelente trabajo!

Has completado el análisis topológico de series temporales.

---

**Autor:** MARK-126
**Licencia:** MIT"""))

# ========== CREATE NOTEBOOK ==========
nb['cells'] = cells

with open('notebooks/05_Series_Temporales_EEG_v2.ipynb', 'w') as f:
    nbf.write(nb, f)

print("✅ Tutorial 5 v2 creado: notebooks/05_Series_Temporales_EEG_v2.ipynb")
