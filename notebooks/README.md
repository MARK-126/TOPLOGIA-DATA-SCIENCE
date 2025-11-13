# Tutoriales Interactivos

Este directorio contiene tutoriales completos de Jupyter Notebook sobre Análisis Topológico de Datos aplicado a Neurociencias.

## 📚 Lista de Tutoriales

### Tutorial 1: Introducción al TDA
**Archivo:** `01_Introduccion_TDA.ipynb`
**Nivel:** Principiante-Intermedio
**Duración:** 90-120 minutos

**Contenido:**
- Conceptos fundamentales de topología
- Complejos simpliciales
- Homología y números de Betti
- Diagramas de persistencia
- Aplicación a redes neuronales sintéticas

**Prerequisitos:** Python básico, NumPy, Matplotlib

---

### Tutorial 2: Homología Persistente Avanzada
**Archivo:** `02_Homologia_Persistente_Avanzada.ipynb`
**Nivel:** Intermedio-Avanzado
**Duración:** 120-150 minutos

**Contenido:**
- Diferentes tipos de filtraciones (Rips, Alpha, Čech)
- Distancias entre diagramas (Bottleneck, Wasserstein)
- Análisis de spike trains
- Características topológicas para ML
- Optimización para grandes datasets

**Prerequisitos:** Tutorial 1, conocimientos de neurociencias básicos

---

### Tutorial 3: Conectividad Cerebral
**Archivo:** `03_Conectividad_Cerebral.ipynb`
**Nivel:** Avanzado
**Duración:** 150-180 minutos

**Contenido:**
- Análisis de conectomas cerebrales
- Redes funcionales vs estructurales
- Detección de comunidades topológicas
- Matrices de correlación fMRI
- Biomarcadores topológicos

**Prerequisitos:** Tutoriales 1 y 2, conocimientos de neuroimagen

---

### Tutorial 4: Algoritmo Mapper
**Archivo:** `04_Mapper_Algorithm.ipynb`
**Nivel:** Avanzado
**Duración:** 120-150 minutos

**Contenido:**
- Algoritmo Mapper: filtro, cover, clustering, nerve
- Visualización de datos de alta dimensión
- Trayectorias de estados cerebrales
- Detección de bifurcaciones neuronales
- Aplicaciones a manifolds neuronales

**Prerequisitos:** Tutoriales 1 y 2, álgebra lineal básica

---

### Tutorial 5: Series Temporales EEG/fMRI
**Archivo:** `05_Series_Temporales_EEG.ipynb`
**Nivel:** Avanzado
**Duración:** 150-180 minutos

**Contenido:**
- Teorema de Takens y embeddings
- Análisis topológico de señales EEG
- Generación y clasificación de estados
- Detección de eventos (crisis epilépticas, sueño)
- Machine learning con características TDA

**Prerequisitos:** Todos los tutoriales anteriores, procesamiento de señales

---

## 🚀 Cómo Usar los Tutoriales

### 1. Instalar Dependencias

```bash
pip install -r ../requirements.txt
```

### 2. Iniciar Jupyter Lab

```bash
jupyter lab
```

### 3. Abrir Tutorial

Navega al tutorial deseado en la interfaz de Jupyter Lab y ejecuta las celdas secuencialmente.

### 4. Ejercicios

Cada tutorial incluye ejercicios prácticos. ¡Completa todos para dominar los conceptos!

---

## 📖 Orden Recomendado

1. **Tutorial 1** - Base fundamental
2. **Tutorial 2** - Técnicas avanzadas
3. **Tutorial 3** - Aplicación a conectividad
4. **Tutorial 4** - Visualización avanzada
5. **Tutorial 5** - Análisis temporal

---

## 💡 Tips

- **Ejecuta todas las celdas:** No te saltes código, cada celda construye sobre la anterior
- **Experimenta:** Modifica parámetros y observa resultados
- **Completa ejercicios:** Son cruciales para el aprendizaje
- **Usa GPU (opcional):** Algunos cálculos se benefician de aceleración GPU

---

## 🆘 Ayuda

Si encuentras errores o tienes preguntas:
1. Revisa la documentación de las bibliotecas
2. Lee los comentarios en el código
3. Abre un issue en el repositorio

---

## 📚 Recursos Adicionales

- [Documentación Ripser](https://ripser.scikit-tda.org/)
- [GUDHI Tutorial](https://gudhi.inria.fr/python/latest/tutorials.html)
- [Giotto-TDA Examples](https://giotto-ai.github.io/gtda-docs/latest/notebooks/index.html)

---

**¡Disfruta aprendiendo TDA aplicado a Neurociencias!** 🧠✨
