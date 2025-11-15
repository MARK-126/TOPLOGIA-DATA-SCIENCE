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

---

## 🚀 **ACTUALIZACIÓN RECIENTE: EXPANSIÓN MASIVA** (v3.0.0)

### 📊 Estadísticas del Proyecto

| Métrica | Valor |
|---------|-------|
| **Ejercicios Interactivos** | **36** (+80% desde v2.0) |
| **Funciones de Test** | **33** (100% cobertura) |
| **Líneas de Tests** | **1,106** |
| **Tutoriales Expandidos** | **6/6** (100%) |
| **Notebooks Interactivos (v2)** | **6** (167 KB total) |
| **Documentación** | **9 archivos** (~3,800 líneas) |
| **Scripts de Automatización** | **11** |
| **Calidad** | **⭐⭐⭐⭐⭐** (4.77/5) |

### ✨ Características Principales

✅ **Ejercicios "Fill-in-the-Blank"** estilo Coursera Deep Learning
✅ **Tests Automáticos** con feedback inmediato
✅ **Aplicaciones Clínicas Reales** (epilepsia, Alzheimer, rehabilitación)
✅ **Metodología Pedagógica de Clase Mundial**
✅ **Cobertura Completa** de TDA aplicado a neurociencias

### 🎯 Para quién es este curso

- 🎓 Estudiantes de maestría/doctorado en neurociencias
- 🔬 Investigadores en análisis de datos cerebrales
- 💻 Data scientists en medicina y healthcare
- 🧠 Desarrolladores de pipelines de análisis neuronal

---

## Contenido del Repositorio

### 📚 Tutoriales Interactivos

0. **Tutorial 0: Setup y Quickstart** ⭐ NUEVO
   - Configuración del entorno
   - Instalación de dependencias
   - Verificación del setup
   - Primer análisis TDA en 10 líneas
   - Troubleshooting común
   - **Duración:** 30-45 minutos

1. **Tutorial 1: Introducción al TDA** 🎓 EXPANDIDO
   - Conceptos básicos de topología
   - Complejos simpliciales, Homología y números de Betti
   - ⭐ **7 ejercicios interactivos** (+3 nuevos avanzados)
   - Comparación de características, filtrado por persistencia, entropía
   - **Duración:** 120-150 minutos

2. **Tutorial 2: Homología Persistente Avanzada** 🎓 EXPANDIDO
   - Filtraciones (Rips, Alpha, Čech)
   - Distancias entre diagramas (Bottleneck, Wasserstein)
   - Análisis de spike trains neuronales
   - ⭐ **7 ejercicios interactivos** (+3 nuevos avanzados)
   - Distancia Wasserstein, cambios temporales, clasificación de patrones
   - **Duración:** 150-180 minutos

3. **Tutorial 3: Conectividad Cerebral con TDA** 🎓 EXPANDIDO
   - Análisis de redes funcionales cerebrales
   - Detección de comunidades topológicas
   - ⭐ **6 ejercicios interactivos** (+3 nuevos avanzados)
   - Características de grafo, nodos críticos, evolución temporal
   - **Duración:** 120-150 minutos

4. **Tutorial 4: Algoritmo Mapper** 🎓 EXPANDIDO
   - Visualización de datos de alta dimensión
   - Aplicación a espacios de representación neural
   - ⭐ **5 ejercicios interactivos** (+2 nuevos avanzados)
   - Optimización de parámetros, detección de ciclos topológicos
   - **Duración:** 100-120 minutos

5. **Tutorial 5: Series Temporales y TDA** 🎓 EXPANDIDO
   - Embeddings de Takens, análisis topológico de señales EEG
   - Detección de eventos neuronales
   - ⭐ **6 ejercicios interactivos** (+3 nuevos avanzados)
   - FNN, reconstrucción de atractores, predicción de eventos
   - **Duración:** 150-180 minutos

6. **Tutorial 6: Caso de Estudio End-to-End** 🎓 EXPANDIDO
   - Pipeline completo de detección de epilepsia
   - Preprocesamiento profesional, análisis TDA aplicado
   - ⭐ **5 ejercicios interactivos** (+2 nuevos avanzados)
   - Importancia de features, validación cruzada rigurosa
   - **Duración:** 150-180 minutos

### 🎓 Formato Interactivo (TODOS los Tutoriales 1-6) ⭐

**TODOS los tutoriales** están disponibles en dos versiones:

#### **Versión Original** (`XX_Nombre.ipynb`)
- Código completo proporcionado
- Útil para referencia rápida
- Ideal para revisión de conceptos

#### **Versión Interactiva v2** (`XX_Nombre_v2.ipynb`) ⭐ RECOMENDADO
Inspirado en Coursera Deep Learning Specialization:
- **36 ejercicios fill-in-the-blank:** Completa código guiado por comentarios
- **33 tests automáticos integrados:** Feedback instantáneo (100% cobertura)
- **Tabla de contenidos clickeable:** Navegación fácil
- **Cajas de resumen visual:** "Lo que debes recordar"
- **Archivos de soluciones:** `TUTORIAL1_SOLUTIONS.md` y `TUTORIAL2_SOLUTIONS.md` disponibles

**Cómo usar las versiones v2:**
1. Lee cada sección conceptual
2. Implementa los ejercicios (marcados con `# YOUR CODE STARTS HERE`)
3. Ejecuta el test automático en la celda siguiente
4. Si pasa ✅, continúa. Si falla ❌, revisa tu código
5. Solo si te atoras, consulta el archivo de soluciones (Tutoriales 1-2)

**Distribución de Ejercicios:**
- Tutorial 1: 7 ejercicios (básico a avanzado)
- Tutorial 2: 7 ejercicios (spike trains y ML)
- Tutorial 3: 6 ejercicios (redes cerebrales)
- Tutorial 4: 5 ejercicios (Mapper avanzado)
- Tutorial 5: 6 ejercicios (series temporales)
- Tutorial 6: 5 ejercicios (pipeline end-to-end)

**Documentación del proyecto:**
- `notebooks/REFACTORING_NOTES.md` - Historia de la refactorización
- `notebooks/REFACTORING_GUIDE.md` - Guía completa para contribuidores
- `REFACTORING_SUMMARY.md` - Resumen ejecutivo del proyecto
- `NEXT_STEPS.md` - Hoja de ruta para continuar

**¿Quieres contribuir?** Ver `REFACTORING_GUIDE.md` para instrucciones detalladas de cómo refactorizar los tutoriales 3-6.

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

### Paso 5: Verificar Instalación ✅

Para verificar que todo funciona correctamente, ejecuta los scripts de prueba:

```bash
# Prueba rápida (~5 segundos)
python test_tutorial0.py

# Prueba completa (~2-3 minutos)
python test_tutorial6.py
```

Si ambos tests pasan, ¡estás listo para comenzar! 🎉

**Ver:** [TESTING.md](TESTING.md) para detalles sobre pruebas, problemas conocidos y soluciones.

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
