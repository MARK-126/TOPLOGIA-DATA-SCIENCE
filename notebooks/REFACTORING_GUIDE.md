# 📘 Guía de Refactorización Interactiva

## Cómo Convertir Tutoriales al Formato Interactivo

Esta guía documenta el proceso para convertir los tutoriales restantes (3-6) al formato interactivo siguiendo el patrón establecido en los Tutoriales 1 y 2.

---

## ✅ Estado Actual de Refactorización

| Tutorial | Estado | Archivo v2 | Tests | Soluciones |
|----------|--------|------------|-------|------------|
| **Tutorial 0** | Original (OK) | N/A | ✅ | N/A |
| **Tutorial 1** | ✅ Completo | `01_Introduccion_TDA_v2.ipynb` | ✅ | `TUTORIAL1_SOLUTIONS.md` |
| **Tutorial 2** | ✅ Completo | `02_Homologia_Persistente_Avanzada_v2.ipynb` | ✅ | `TUTORIAL2_SOLUTIONS.md` |
| **Tutorial 3** | ⏳ Pendiente | - | - | - |
| **Tutorial 4** | ⏳ Pendiente | - | - | - |
| **Tutorial 5** | ⏳ Pendiente | - | - | - |
| **Tutorial 6** | ⏳ Pendiente | - | - | - |

---

## 📋 Patrón de Refactorización

### Paso 1: Identificar Funciones Clave

Lee el tutorial original e identifica 3-5 funciones que:
- Sean conceptualmente importantes
- Tengan lógica no trivial (no solo llamadas a bibliotecas)
- Puedan dividirse en pasos claros
- Enseñen habilidades transferibles

**Ejemplo (Tutorial 2):**
- ✅ `generate_brain_state_realistic` - Generación de datos
- ✅ `generate_spike_trains` - Simulación neuronal
- ✅ `spike_trains_to_state_space` - Transformación de datos
- ✅ `extract_topological_features` - Análisis de resultados

### Paso 2: Crear Estructura del Notebook

Usa el siguiente template:

```python
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

# 1. Título y objetivos
cells.append(nbf.v4.new_markdown_cell("""# Tutorial X: Título

## Subtítulo (Versión Interactiva)

**Autor:** MARK-126
**Nivel:** ...
**Tiempo estimado:** ...

---

## 🎯 Objetivos de Aprendizaje

1. ✅ ...
2. ✅ ...

---

## ⚠️ Nota Importante sobre Ejercicios

Este notebook contiene **N ejercicios interactivos**...
"""))

# 2. Tabla de contenidos
cells.append(nbf.v4.new_markdown_cell("""<a name='toc'></a>
## 📚 Tabla de Contenidos

- [1 - Setup](#1)
- [2 - Conceptos](#2)
- ...
"""))

# 3. Setup
cells.append(nbf.v4.new_markdown_cell("""<a name='1'></a>
## 1 - Setup

[Volver al índice](#toc)"""))

cells.append(nbf.v4.new_code_cell("""# Importaciones
import numpy as np
...
from tda_tests import test_funcion1, test_funcion2
print("✅ Setup completado")"""))

# 4. Para cada sección:
#    - Markdown con teoría
#    - Ejercicio fill-in-the-blank
#    - Test automático
#    - Visualización
#    - Caja de resumen

# 5. Resumen final

nb['cells'] = cells
```

### Paso 3: Convertir Función a Ejercicio

**Original:**
```python
def mi_funcion(parametros):
    # Código completo
    paso1 = calcular_algo()
    paso2 = procesar(paso1)
    return paso2
```

**Versión Interactiva:**
```python
def mi_funcion(parametros):
    \"\"\"
    Descripción clara de qué hace.

    Arguments:
    parametros -- descripción

    Returns:
    resultado -- descripción
    \"\"\"
    # Paso 1: Descripción del paso
    # (approx. X lines)
    # YOUR CODE STARTS HERE


    # YOUR CODE ENDS HERE

    # Paso 2: Descripción del paso
    # (approx. Y lines)
    # YOUR CODE STARTS HERE

    # YOUR CODE ENDS HERE

    return resultado
```

### Paso 4: Crear Tests

En `tda_tests.py`, agrega:

```python
def test_mi_funcion(target):
    """
    Test para mi_funcion
    """
    print("Ejecutando tests para mi_funcion...")

    # Test 1: Caso básico
    resultado = target(param1, param2)
    assert condición1, "❌ Mensaje de error específico"

    # Test 2: Caso edge
    resultado2 = target(param_edge)
    assert condición2, "❌ Mensaje de error específico"

    # Test 3: Caso límite
    resultado3 = target(param_limite)
    assert condición3, "❌ Mensaje de error específico"

    print("\033[92m✅ Todos los tests de mi_funcion pasaron!\033[0m")
```

### Paso 5: Escribir Soluciones

En `TUTORIALX_SOLUTIONS.md`:

```markdown
## Ejercicio N: nombre_funcion

### Solución Completa:

```python
def mi_funcion(parametros):
    # Código completo con comentarios explicativos
    ...
```

### Explicación Paso a Paso:

#### **Paso 1: Descripción**
- **Objetivo:** Qué logra este paso
- **Método:** Cómo lo hace
- Explicación línea por línea si es complejo

**Intuición neurobiológica/matemática:** Por qué esto tiene sentido

#### **Paso 2: ...**
...

### Consejos de Debugging:

**Error común 1:** Descripción
- **Causa:** Por qué ocurre
- **Solución:** Cómo arreglarlo

**Error común 2:** ...
```

### Paso 6: Agregar Cajas de Resumen

Después de cada sección principal:

```markdown
<div style="background-color:#e3f2fd; padding:15px; border-left:5px solid #2196f3; margin: 20px 0;">

**💡 Lo que debes recordar:**

- Punto clave 1
- Punto clave 2
- Punto clave 3

</div>
```

---

## 🎨 Elementos de Diseño

### Iconos y Emojis

Usa consistentemente:
- 🎯 Objetivos
- ✅ Completado/Correcto
- ❌ Error
- ⚠️ Advertencia
- 💡 Insight/Tip
- 🧠 Neurociencia
- 📊 Datos/Gráficos
- 🔬 Análisis
- 🔥 Spike trains
- ⏳ En progreso
- 🎉 Felicitaciones

### Estilos de Cajas

**Recordatorios (Azul):**
```markdown
<div style="background-color:#e3f2fd; padding:15px; border-left:5px solid #2196f3;">
```

**Interpretación Neuronal (Amarillo):**
```markdown
<div style="background-color:#fff3cd; padding:15px; border-left:5px solid #ffc107;">
```

**Observaciones Importantes (Verde):**
```markdown
<div style="background-color:#e8f5e9; padding:15px; border-left:5px solid #4caf50;">
```

**Felicitaciones (Morado):**
```markdown
<div style="background-color:#f3e5f5; padding:15px; border-left:5px solid #9c27b0;">
```

---

## 📝 Checklist de Calidad

Antes de considerar un tutorial "completo", verifica:

### Contenido
- [ ] 3-5 ejercicios fill-in-the-blank
- [ ] Tabla de contenidos clickeable
- [ ] Importaciones de tests incluidas
- [ ] Cajas de resumen después de cada sección
- [ ] Visualizaciones embebidas
- [ ] Outputs esperados mostrados

### Tests
- [ ] Función de test para cada ejercicio
- [ ] Al menos 3 casos de test por función
- [ ] Mensajes de error específicos
- [ ] Casos edge/límite cubiertos
- [ ] Tests agregados a `run_all_tests_tutorialX()`

### Soluciones
- [ ] Código completo y comentado
- [ ] Explicación paso a paso
- [ ] Intuición biológica/matemática
- [ ] Al menos 3 errores comunes documentados
- [ ] Ejercicios adicionales (desafíos opcionales)

### Formato
- [ ] Anchors HTML para navegación
- [ ] Código formateado consistentemente
- [ ] Markdown limpio (sin errores de sintaxis)
- [ ] Imágenes optimizadas (si aplica)
- [ ] Metadatos actualizados (autor, fecha)

---

## 🛠️ Herramientas Útiles

### Script de Generación de Notebooks

Ver `create_tutorial2_v2.py` como template.

Modificaciones necesarias:
1. Cambiar título y metadatos
2. Actualizar lista de tests importados
3. Ajustar contenido de celdas
4. Actualizar ejercicios

### Testing Local

Antes de commit:
```bash
cd notebooks
jupyter nbconvert --to notebook --execute TUTORIALX_v2.ipynb
```

Esto verifica que el notebook se ejecuta sin errores.

### Validación de Markdown

```bash
pip install mdformat
mdformat TUTORIALX_SOLUTIONS.md
```

---

## 📊 Estimación de Tiempo

Por tutorial:
- **Análisis del original:** 30-45 min
- **Identificar ejercicios:** 15-20 min
- **Crear script de generación:** 45-60 min
- **Escribir tests:** 30-45 min
- **Documentar soluciones:** 60-90 min
- **Testing y debugging:** 30-45 min

**Total por tutorial:** ~4-5 horas

---

## 🚀 Próximos Pasos Recomendados

### Prioridad Alta
1. **Tutorial 3** (Conectividad Cerebral): Análisis de redes
2. **Tutorial 6** (Caso de Estudio): Aplicación end-to-end

### Prioridad Media
3. **Tutorial 4** (Mapper): Visualización
4. **Tutorial 5** (Series Temporales): Análisis temporal

### Ejercicios Candidatos por Tutorial

**Tutorial 3:**
- `build_connectivity_matrix`
- `detect_communities_topological`
- `compare_functional_vs_structural`

**Tutorial 4:**
- `mapper_graph_construction`
- `choose_filter_function`
- `visualize_mapper_interactive`

**Tutorial 5:**
- `takens_embedding`
- `sliding_window_persistence`
- `detect_events_from_topology`

**Tutorial 6:**
- `preprocess_eeg_clinical`
- `extract_comprehensive_features`
- `train_topological_classifier`

---

## 🤝 Contribuciones

Si quieres contribuir refactorizando un tutorial:

1. **Fork** el repositorio
2. **Crea branch** `feature/tutorial-X-interactive`
3. Sigue este guía paso a paso
4. **Test localmente** antes de commit
5. **Submit PR** con descripción detallada

### Template de PR

```markdown
## Tutorial X Versión Interactiva

### Cambios realizados:
- ✅ Creado `0X_Nombre_v2.ipynb`
- ✅ Agregados N ejercicios fill-in-the-blank
- ✅ Tests en `tda_tests.py`
- ✅ Soluciones en `TUTORIALX_SOLUTIONS.md`

### Ejercicios incluidos:
1. `funcion1` - Descripción
2. `funcion2` - Descripción
...

### Testing:
- [x] Notebook ejecuta sin errores
- [x] Todos los tests pasan
- [x] Soluciones verificadas
```

---

## 📚 Referencias

- **Tutorial 1 v2:** Ejemplo simple y claro
- **Tutorial 2 v2:** Ejemplo más complejo con múltiples patrones
- **Coursera DL Spec:** Inspiración para ejercicios
- **Fast.ai:** Filosofía de aprendizaje activo

---

**Última actualización:** 2025-01-15
**Autor:** MARK-126
**Versión:** 1.0
