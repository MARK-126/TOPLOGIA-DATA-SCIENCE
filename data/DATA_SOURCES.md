# Fuentes de Datos Reales para TDA + Neurociencias

Este documento describe cómo obtener y usar datasets reales de neurociencias para los tutoriales.

---

## 📊 Datasets Incluidos (Sintéticos para Práctica)

Los tutoriales 1-5 incluyen **generación de datos sintéticos** que simulan:
- Redes neuronales
- Spike trains
- Señales EEG
- Conectividad fMRI

Estos son ideales para aprender sin preocuparte por el formato de datos reales.

---

## 🌐 Datasets Públicos Recomendados

### 1. **PhysioNet - CHB-MIT Scalp EEG Database**

**Descripción:** EEG de pacientes pediátricos con epilepsia
**URL:** https://physionet.org/content/chbmit/1.0.0/
**Tamaño:** ~3 GB
**Formato:** .edf (European Data Format)

**Uso en Tutorial 6**

**Descargar manualmente:**
```bash
cd TOPLOGIA-DATA-SCIENCE/data/raw/
wget -r -N -c -np https://physionet.org/files/chbmit/1.0.0/chb01/
```

**O usar la herramienta incluida:**
```python
from src.neuro_utils.data_loader import download_chbmit
download_chbmit('data/raw/', subject='chb01')
```

**Características:**
- 23 canales de EEG
- Frecuencia de muestreo: 256 Hz
- Incluye anotaciones de crisis epilépticas
- Perfecto para clasificación binaria (ictal vs interictal)

---

### 2. **Human Connectome Project (HCP)**

**Descripción:** Datos de neuroimagen de 1200+ sujetos sanos
**URL:** https://www.humanconnectome.org/
**Tamaño:** Variable (100 MB - 1 TB por sujeto)
**Formato:** NIFTI, CIFTI

**Uso:** Tutorial 3 (Conectividad Cerebral)

**Requisitos:**
- Registro gratuito en HCP
- Aceptar términos de uso
- Usar AWS S3 o Aspera para descarga

**Datos mínimos recomendados:**
```
Subject_100307/
├── rfMRI_REST1_LR.nii.gz  # fMRI en reposo
└── T1w.nii.gz              # Anatomía
```

**Uso en Python:**
```python
from nilearn import datasets
hcp_dataset = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
```

---

### 3. **OpenNeuro - Multiple Studies**

**Descripción:** Repositorio abierto de neuroimagen
**URL:** https://openneuro.org/
**Tamaño:** Variable
**Formato:** BIDS (Brain Imaging Data Structure)

**Datasets recomendados:**

#### ds003097 - Emotional faces
- fMRI durante tarea cognitiva
- ~20 sujetos
- Ideal para análisis de conectividad

#### ds000030 - UCLA Consortium
- fMRI multi-modal
- Incluye controles y pacientes
- Perfecto para comparaciones

**Descargar con DataLad:**
```bash
pip install datalad
datalad install https://github.com/OpenNeuroDatasets/ds003097
cd ds003097
datalad get sub-01/func/
```

---

### 4. **MNE-Python Sample Data**

**Descripción:** Datos de ejemplo de MEG/EEG
**URL:** Incluido en MNE-Python
**Tamaño:** ~1.5 GB
**Formato:** FIF

**Descargar automáticamente:**
```python
import mne
sample_data_folder = mne.datasets.sample.data_path()
print(sample_data_folder)
```

**Contenido:**
- EEG/MEG de experimento visual-auditivo
- Anatomía MRI
- Source space
- Forward solution

**Uso:** Tutorial 5 (Series Temporales)

---

### 5. **Alzheimer's Disease Neuroimaging Initiative (ADNI)**

**Descripción:** fMRI, PET, MRI de pacientes con Alzheimer
**URL:** http://adni.loni.usc.edu/
**Tamaño:** TB
**Formato:** DICOM, NIFTI

**Requisitos:**
- Solicitud de acceso (aprobación en ~1 semana)
- Afiliación académica

**Uso potencial:**
- Comparación topológica: sanos vs Alzheimer
- Progresión de enfermedad
- Biomarcadores

---

## 🔧 Herramientas de Descarga

### Script Incluido: `download_data.py`

Ubicación: `data/download_data.py`

**Uso:**
```bash
python data/download_data.py --dataset chbmit --subject chb01
python data/download_data.py --dataset mne-sample
```

**Datasets soportados:**
- `chbmit`: CHB-MIT EEG (epilepsia)
- `mne-sample`: MNE sample data
- `hcp-test`: HCP subset pequeño (requiere credenciales)

---

## 📁 Estructura de Directorios

```
data/
├── raw/                    # Datos sin procesar (descargas)
│   ├── chbmit/
│   │   └── chb01/
│   ├── mne_sample/
│   └── hcp/
│
├── processed/              # Datos preprocesados
│   ├── eeg_features.csv
│   ├── connectivity_matrices.npy
│   └── persistence_diagrams.pkl
│
├── external/               # Enlaces simbólicos a datos externos
│
└── DATA_SOURCES.md         # Este archivo
```

---

## 🚀 Quick Start con Datos Reales

### Opción 1: Dataset Pequeño (MNE Sample)

```python
# Descarga automática (~1.5 GB, una sola vez)
import mne
mne.datasets.sample.data_path()

# Usar en Tutorial 5
```

### Opción 2: EEG de Epilepsia (CHB-MIT)

```bash
# Descargar UN sujeto (~100 MB)
cd data/raw/
wget https://physionet.org/files/chbmit/1.0.0/chb01/chb01_03.edf
```

```python
# Usar en Tutorial 6
import mne
raw = mne.io.read_raw_edf('data/raw/chb01/chb01_03.edf', preload=True)
```

### Opción 3: Conectividad Simulada Realista

Si no quieres descargar datos pesados, usa los generadores incluidos:

```python
from src.neuro_utils.data_generator import generate_realistic_fmri
fmri_data = generate_realistic_fmri(n_rois=90, n_timepoints=200)
```

---

## ⚖️ Consideraciones Éticas y Legales

### Datos Públicos:
- ✅ PhysioNet: Uso libre con atribución
- ✅ MNE Sample: MIT License
- ✅ OpenNeuro: Varía por dataset (verificar)

### Datos Restringidos:
- ⚠️ HCP: Requiere registro
- ⚠️ ADNI: Requiere aprobación
- ⚠️ UK Biobank: Requiere afiliación académica

### Reglas Generales:
1. **Citar siempre** la fuente de datos
2. **Respetar términos de uso**
3. **No redistribuir** datos sin permiso
4. **Anonimizar** cualquier dato propio

---

## 📚 Formato de Citas

### PhysioNet CHB-MIT:
```
Goldberger, A., et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet:
Components of a new research resource for complex physiologic signals.
Circulation [Online]. 101 (23), pp. e215–e220.
```

### MNE Sample:
```
Gramfort, A., et al. (2013). MEG and EEG data analysis with MNE-Python.
Frontiers in Neuroscience, 7(267), 1-13.
```

### HCP:
```
Van Essen, D.C., et al. (2013). The WU-Minn Human Connectome Project:
An overview. NeuroImage, 80, 62-79.
```

---

## ❓ FAQ

**Q: ¿Cuánto espacio necesito?**
A: Mínimo 5 GB. Recomendado 20 GB para trabajar cómodamente.

**Q: ¿Puedo usar mis propios datos?**
A: ¡Sí! Los tutoriales están diseñados para ser adaptables.

**Q: ¿Necesito descargar todos los datasets?**
A: No. Empieza con MNE sample (Tutorial 5) o datos sintéticos.

**Q: Mi universidad tiene datos, ¿puedo usarlos?**
A: Sí, con aprobación del comité de ética y consentimiento.

---

## 🆘 Soporte

Si tienes problemas descargando datos:
1. Verifica tu conexión a internet
2. Revisa espacio en disco
3. Consulta la documentación oficial del dataset
4. Abre un issue en el repositorio

---

**Última actualización:** 2025-01-13
**Autor:** MARK-126
**Licencia:** MIT
