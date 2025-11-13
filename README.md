# Topología en Data Science: Aplicaciones a Neurociencias

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Descripción

Este repositorio contiene **tutoriales interactivos completos** que exploran la aplicación de **Análisis Topológico de Datos (TDA)** en el campo de las **neurociencias**. Cada tutorial combina teoría matemática rigurosa, implementaciones prácticas en Python, y aplicaciones reales a datos neurocientíficos.

### ¿Qué es el Análisis Topológico de Datos (TDA)?

El TDA es un campo emergente que utiliza conceptos de topología algebraica para analizar la "forma" de los datos. A diferencia de métodos estadísticos tradicionales, el TDA puede capturar:
- Estructura global de conjuntos de datos complejos
- Patrones no lineales
- Características topológicas persistentes (huecos, componentes conectadas, cavidades)
- Invariantes bajo transformaciones continuas

### ¿Por qué TDA en Neurociencias?

El cerebro es un sistema complejo con propiedades topológicas fascinantes:
- **Redes neuronales** con topologías complejas
- **Conectividad cerebral** que forma grafos de alta dimensión
- **Señales temporales** (EEG, fMRI) con patrones topológicos
- **Espacios de representación neural** con estructura geométrica

## Contenido del Repositorio

### 📚 Tutoriales Interactivos

0. **Tutorial 0: Setup y Quickstart** ⭐ NUEVO
   - Configuración del entorno
   - Instalación de dependencias
   - Verificación del setup
   - Primer análisis TDA en 10 líneas
   - Troubleshooting común
   - **Duración:** 30-45 minutos

1. **Tutorial 1: Introducción al TDA**
   - Conceptos básicos de topología
   - Complejos simpliciales
   - Homología y números de Betti
   - Ejemplos con datos sintéticos
   - **Duración:** 90-120 minutos

2. **Tutorial 2: Homología Persistente**
   - Filtraciones y diagramas de persistencia
   - Cálculo de características topológicas
   - Aplicación a patrones de activación neuronal
   - Análisis de estabilidad

3. **Tutorial 3: Conectividad Cerebral con TDA**
   - Análisis de redes funcionales cerebrales
   - Detección de comunidades topológicas
   - Métricas de conectividad basadas en TDA
   - Ejemplos con datos de fMRI

4. **Tutorial 4: Algoritmo Mapper**
   - Visualización de datos de alta dimensión
   - Aplicación a espacios de representación neural
   - Análisis de estados cerebrales
   - Clustering topológico

5. **Tutorial 5: Series Temporales y TDA**
   - Embeddings de Takens
   - Análisis topológico de señales EEG
   - Detección de eventos neuronales
   - Clasificación de estados cognitivos
   - **Duración:** 150-180 minutos

6. **Tutorial 6: Caso de Estudio End-to-End** ⭐ NUEVO (En desarrollo)
   - Pipeline completo con datos reales
   - Detección de epilepsia con EEG (PhysioNet)
   - Preprocesamiento profesional
   - Análisis TDA aplicado
   - Machine learning y evaluación
   - Interpretación clínica
   - **Duración:** 180+ minutos

### 📊 Datos Reales

Ver `data/DATA_SOURCES.md` para instrucciones de descarga de:
- **PhysioNet CHB-MIT:** EEG de epilepsia
- **Human Connectome Project:** fMRI
- **OpenNeuro:** Múltiples estudios
- **MNE Sample Data:** MEG/EEG

### 🛠️ Estructura del Proyecto

```
TOPLOGIA-DATA-SCIENCE/
├── notebooks/           # Tutoriales Jupyter interactivos
├── tutorials/           # Versiones HTML de los tutoriales
├── src/                # Código fuente reutilizable
│   ├── tda_tools/      # Herramientas de TDA
│   ├── neuro_utils/    # Utilidades de neurociencias
│   └── visualization/  # Funciones de visualización
├── tests/              # Tests unitarios
├── data/               # Datos de ejemplo
│   ├── raw/            # Datos sin procesar
│   └── processed/      # Datos procesados
├── docs/               # Documentación adicional
└── assets/             # Recursos (imágenes, diagramas)
```

## Instalación

### Requisitos Previos

- Python 3.8 o superior
- pip o conda

### Paso 1: Clonar el repositorio

```bash
git clone https://github.com/MARK-126/TOPLOGIA-DATA-SCIENCE.git
cd TOPLOGIA-DATA-SCIENCE
```

### Paso 2: Crear entorno virtual (recomendado)

```bash
# Con venv
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# O con conda
conda create -n tda-neuro python=3.10
conda activate tda-neuro
```

### Paso 3: Instalar dependencias

```bash
pip install -r requirements.txt
```

### Paso 4: Iniciar Jupyter

```bash
jupyter lab
```

## Uso Rápido

### Ejemplo: Análisis Topológico de Red Neuronal

```python
from src.tda_tools import compute_persistence_diagram
from src.neuro_utils import load_connectivity_matrix
from src.visualization import plot_persistence_diagram

# Cargar matriz de conectividad
connectivity = load_connectivity_matrix('data/processed/brain_network.npy')

# Calcular diagrama de persistencia
diagram = compute_persistence_diagram(connectivity)

# Visualizar
plot_persistence_diagram(diagram)
```

### Ejemplo: Análisis de Señal EEG

```python
from src.tda_tools import sliding_window_embedding
from src.neuro_utils import load_eeg_data

# Cargar datos EEG
eeg_signal = load_eeg_data('data/raw/eeg_sample.h5')

# Crear embedding topológico
embedding = sliding_window_embedding(eeg_signal, window=100, stride=10)

# Analizar topología
persistence = compute_persistence_diagram(embedding)
```

## Datasets Incluidos

- **Datos sintéticos**: Redes neuronales simuladas
- **EEG público**: Señales de ejemplo (dataset Physionet)
- **Conectividad fMRI**: Matrices de correlación funcional
- **Patrones de activación**: Datos de spike trains sintéticos

## Bibliotecas Principales

- **[Giotto-TDA](https://giotto-ai.github.io/gtda-docs/)**: Suite completa de TDA
- **[Ripser](https://ripser.scikit-tda.org/)**: Cálculo rápido de homología persistente
- **[GUDHI](https://gudhi.inria.fr/)**: Biblioteca robusta de geometría computacional
- **[MNE-Python](https://mne.tools/)**: Análisis de señales EEG/MEG
- **[Nilearn](https://nilearn.github.io/)**: Machine learning para neuroimaging
- **[NetworkX](https://networkx.org/)**: Análisis de redes complejas

## Temas Avanzados

- Análisis multiescala de redes cerebrales
- Aprendizaje topológico (Topological Machine Learning)
- Mapper interactivo para datos neuronales
- TDA en espacios de representación de redes profundas
- Análisis dinámico de conectividad funcional

## Recursos Adicionales

### Libros y Papers
- Carlsson, G. (2009). "Topology and data"
- Edelsbrunner & Harer (2010). "Computational Topology"
- Giusti et al. (2015). "Clique topology reveals intrinsic structure in neural correlations"
- Curto (2017). "What can topology tell us about the neural code?"

### Cursos Online
- Applied Algebraic Topology (Stanford)
- Topological Data Analysis (Coursera)
- Computational Neuroscience (Coursera)

## Contribuciones

¡Las contribuciones son bienvenidas! Por favor:

1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/nueva-caracteristica`)
3. Commit tus cambios (`git commit -m 'Agregar nueva característica'`)
4. Push a la rama (`git push origin feature/nueva-caracteristica`)
5. Abre un Pull Request

## Licencia

Este proyecto está licenciado bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.

## Autores

- **MARK-126** - [GitHub](https://github.com/MARK-126)

## Agradecimientos

- Comunidad de TDA
- Investigadores en neurociencias computacionales
- Desarrolladores de las bibliotecas open-source utilizadas

## Contacto

Para preguntas, sugerencias o colaboraciones, abre un issue en este repositorio.

---

**¡Explora la topología del cerebro!** 🧠✨
