# 🎨 Imágenes Explicativas para Tutoriales de TDA

Este directorio contiene diagramas y visualizaciones de alta calidad para los tutoriales de Análisis Topológico de Datos.

---

## 📊 Catálogo de Imágenes

### 1. **simplicial_construction_steps.png** (108 KB)
**Uso:** Tutorial 1 - Sección 3 (Complejos Simpliciales)

**Contenido:** Muestra la construcción paso a paso de un complejo de Vietoris-Rips con tres valores de epsilon (0.5, 1.0, 1.5).

**Conceptos ilustrados:**
- Cómo el parámetro ε controla la densidad del complejo
- Progresión de 0-simplejos (puntos) → 1-simplejos (aristas) → 2-simplejos (triángulos)
- Mayor ε = más conexiones = estructura más densa

**Cómo usar en notebook:**
```markdown
![Construcción de Complejo Simplicial](images/simplicial_construction_steps.png)
```

---

### 2. **persistence_diagram_anatomy.png** (193 KB)
**Uso:** Tutorial 1 - Sección 4 (Homología Persistente)

**Contenido:** Anatomía detallada de un diagrama de persistencia con anotaciones explicativas.

**Conceptos ilustrados:**
- Ejes: Birth (nacimiento) vs Death (muerte)
- Línea diagonal: referencia para medir persistencia
- Features persistentes (lejos de diagonal) vs ruido (cerca de diagonal)
- Cómo calcular lifetime (persistencia)
- Diferencia entre H₀ (rojo) y H₁ (azul)

**Cómo usar en notebook:**
```markdown
![Anatomía de Diagrama de Persistencia](images/persistence_diagram_anatomy.png)

**Regla de oro:** Puntos lejos de la diagonal = características importantes
```

---

### 3. **betti_numbers_evolution.png** (159 KB)
**Uso:** Tutorial 1 - Sección 4 (Números de Betti)

**Contenido:** Tres paneles mostrando evolución de números de Betti para un círculo.

**Conceptos ilustrados:**
- Panel 1: Datos originales (círculo con ruido)
- Panel 2: β₀ converge a 1 (una componente conectada)
- Panel 3: β₁ = 1 detecta el círculo
- Cómo cambia la topología al aumentar ε

**Cómo usar en notebook:**
```markdown
![Evolución de Números de Betti](images/betti_numbers_evolution.png)

**Observa cómo β₁ = 1 persiste en un rango de ε, indicando la presencia robusta del círculo.**
```

---

### 4. **homology_dimensions_comparison.png** (172 KB)
**Uso:** Tutorial 1 - Sección 4 (Conceptos de Homología)

**Contenido:** Comparación visual de las tres dimensiones de homología.

**Conceptos ilustrados:**
- **H₀:** Componentes conectadas (ejemplo: 3 grupos separados)
- **H₁:** Ciclos/loops (ejemplo: un círculo)
- **H₂:** Cavidades (ejemplo: una esfera hueca)
- Interpretación intuitiva de cada dimensión

**Cómo usar en notebook:**
```markdown
![Comparación de Dimensiones](images/homology_dimensions_comparison.png)

**Analogía:** H₀ cuenta islas, H₁ cuenta lagos, H₂ cuenta burbujas.
```

---

### 5. **persistence_concept.png** (633 KB)
**Uso:** Tutorial 1 - Sección 7 (Homología Persistente Avanzada)

**Contenido:** Secuencia de 6 imágenes mostrando cómo evolucionan las características con diferentes ε.

**Conceptos ilustrados:**
- Ruido (puntos rojos) vs señal (círculo azul)
- ε bajo: muchas componentes (ruido y señal separados)
- ε medio: el círculo se forma (característica persistente)
- ε alto: todo se conecta (características desaparecen)
- **Concepto clave:** Features persistentes sobreviven a través de múltiples escalas

**Cómo usar en notebook:**
```markdown
![Concepto de Persistencia](images/persistence_concept.png)

**Persistencia = Robustez:** Las características que aparecen en muchas escalas de ε son las verdaderamente importantes.
```

---

## 🎯 Cómo Usar estas Imágenes

### En Jupyter Notebooks:

```markdown
## Construcción de Complejo Simplicial

Observa cómo aumentar ε conecta más puntos:

![](images/simplicial_construction_steps.png)
```

### En Markdown (GitHub):

```markdown
![Título descriptivo](../notebooks/images/nombre_imagen.png)
```

### En HTML (notebooks con formato personalizado):

```html
<img src="images/nombre_imagen.png" alt="Descripción" width="800"/>
```

---

## 🔄 Regenerar Imágenes

Si necesitas regenerar las imágenes (por ejemplo, cambiar colores, tamaños, o contenido):

```bash
cd notebooks
python3 generate_tutorial_images.py
```

**Script:** `generate_tutorial_images.py` contiene todas las funciones de generación.

**Personalización:**
- Cambia colores en las líneas con códigos hex (ej: `#2196f3`)
- Ajusta DPI en `plt.savefig(..., dpi=150)`
- Modifica tamaños de figura en `figsize=(ancho, alto)`

---

## 📐 Especificaciones Técnicas

| Imagen | Dimensiones (aprox) | Formato | DPI | Uso de color |
|--------|---------------------|---------|-----|--------------|
| simplicial_construction_steps.png | 1800×600 px | PNG | 150 | Material Design |
| persistence_diagram_anatomy.png | 1200×1200 px | PNG | 150 | Anotaciones multicolor |
| betti_numbers_evolution.png | 1800×500 px | PNG | 150 | Rojo (β₀), Azul (β₁) |
| homology_dimensions_comparison.png | 1800×600 px | PNG | 150 | Rojo, Verde, Azul |
| persistence_concept.png | 1800×1200 px | PNG | 150 | Azul (señal), Rojo (ruido) |

**Paleta de colores usada:**
- Rojo: `#f44336` (H₀, componentes)
- Azul: `#2196f3` (H₁, ciclos)
- Verde: `#4caf50` (datos secundarios)
- Morado: `#9c27b0` (H₂, cavidades)
- Amarillo: `#ffc107` (anotaciones, advertencias)

---

## 🎨 Mejoras Futuras

### Fase 3 (Planeado):
- [ ] Crear GIFs animados mostrando evolución de epsilon
- [ ] Diagramas interactivos (widgets de Jupyter)
- [ ] Más ejemplos de aplicaciones neurocientíficas
- [ ] Comparaciones lado a lado de diferentes datasets

### Contribuciones:

¿Tienes ideas para nuevas visualizaciones?

1. Edita `generate_tutorial_images.py`
2. Agrega una nueva función `generate_[concepto]()`
3. Llama la función en el `if __name__ == "__main__"`
4. Ejecuta el script
5. Documenta la nueva imagen aquí
6. Submit PR

---

## 📚 Referencias

**Herramientas usadas:**
- **Matplotlib:** Todas las visualizaciones
- **NumPy/SciPy:** Cálculos y generación de datos
- **Ripser:** Homología persistente

**Inspiración de diseño:**
- Material Design color palette
- "Visual Group Theory" by Nathan Carter
- Coursera Deep Learning Specialization images
- Topological Data Analysis (Gunnar Carlsson)

---

## 📄 Licencia

Estas imágenes están bajo la misma licencia MIT que el repositorio.

- ✅ Uso libre en contextos educativos
- ✅ Modificación permitida
- ✅ Redistribución permitida
- ⚠️ Atribución apreciada (pero no requerida)

---

## 📬 Contacto

¿Preguntas sobre las imágenes o cómo usarlas?

- Abre un issue en el repositorio
- Revisa `generate_tutorial_images.py` para detalles técnicos
- Consulta los tutoriales para ver ejemplos de uso

---

**¡Disfruta de las visualizaciones!** 🎨📊✨

*Actualizado: 2024-11-13*
