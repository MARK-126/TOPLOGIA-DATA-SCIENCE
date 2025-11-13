#!/usr/bin/env python3
"""
Script de prueba para verificar el código del Tutorial 6
"""

import numpy as np
from scipy import signal
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from ripser import ripser

print("=" * 60)
print("PRUEBA DEL TUTORIAL 6: Caso de Estudio Epilepsia")
print("=" * 60)

# 1. Test: Generación de datos EEG realistas
print("\n1. Generando datos EEG sintéticos...")
def generate_realistic_eeg_segment(duration=10, fs=256, state='interictal', n_channels=23):
    """Genera un segmento de EEG realista"""
    n_samples = int(duration * fs)
    t = np.linspace(0, duration, n_samples)
    eeg_data = np.zeros((n_channels, n_samples))

    for ch in range(n_channels):
        # Ruido base
        noise = np.random.randn(n_samples) * 0.5

        # Ritmos cerebrales normales
        delta = 0.8 * np.sin(2 * np.pi * np.random.uniform(1, 4) * t)
        theta = 0.6 * np.sin(2 * np.pi * np.random.uniform(4, 8) * t)
        alpha = 1.2 * np.sin(2 * np.pi * np.random.uniform(8, 13) * t)
        beta = 0.4 * np.sin(2 * np.pi * np.random.uniform(13, 30) * t)

        if state == 'ictal':
            # Actividad epiléptica
            seizure_freq = np.random.uniform(3, 5)
            spike_wave = 2.5 * np.sin(2 * np.pi * seizure_freq * t)
            harmonics = 0.8 * np.sin(2 * np.pi * 2 * seizure_freq * t)
            hfo = 0.3 * np.sin(2 * np.pi * np.random.uniform(80, 200) * t)

            eeg_data[ch, :] = spike_wave + harmonics + hfo + noise * 0.3
        else:
            eeg_data[ch, :] = delta + theta + alpha + beta + noise

    return eeg_data

# Generar datos de prueba
interictal = generate_realistic_eeg_segment(duration=5, state='interictal')
ictal = generate_realistic_eeg_segment(duration=5, state='ictal')
print(f"   Interictal shape: {interictal.shape}")
print(f"   Ictal shape: {ictal.shape}")
print("   ✓ Generación exitosa")

# 2. Test: Preprocesamiento
print("\n2. Probando pipeline de preprocesamiento...")
def preprocess_eeg(eeg_data, fs=256):
    """Pipeline completo de preprocesamiento"""
    processed = eeg_data.copy()
    n_channels, n_samples = processed.shape

    # 1. Bandpass filter (0.5-50 Hz)
    sos_bp = signal.butter(4, [0.5, 50], btype='band', fs=fs, output='sos')
    for ch in range(n_channels):
        processed[ch, :] = signal.sosfilt(sos_bp, processed[ch, :])

    # 2. Notch filter (60 Hz)
    b_notch, a_notch = signal.iirnotch(60, 30, fs)
    for ch in range(n_channels):
        processed[ch, :] = signal.filtfilt(b_notch, a_notch, processed[ch, :])

    # 3. Common Average Reference (CAR)
    avg_signal = np.mean(processed, axis=0)
    processed = processed - avg_signal

    # 4. Z-score normalization
    for ch in range(n_channels):
        processed[ch, :] = (processed[ch, :] - np.mean(processed[ch, :])) / (np.std(processed[ch, :]) + 1e-10)

    return processed

processed_interictal = preprocess_eeg(interictal)
processed_ictal = preprocess_eeg(ictal)
print(f"   Processed interictal: {processed_interictal.shape}")
print(f"   Mean: {np.mean(processed_interictal):.6f}, Std: {np.std(processed_interictal):.6f}")
print("   ✓ Preprocesamiento exitoso")

# 3. Test: Takens embedding
print("\n3. Probando Takens embedding...")
def takens_embedding(signal_data, delay, dimension):
    """Crear embedding de Takens"""
    n = len(signal_data)
    m = n - (dimension - 1) * delay
    embedded = np.zeros((m, dimension))
    for i in range(dimension):
        start = i * delay
        embedded[:, i] = signal_data[start:start+m]
    return embedded

test_signal = processed_interictal[0, :]
embedded = takens_embedding(test_signal, delay=10, dimension=3)
print(f"   Signal length: {len(test_signal)}")
print(f"   Embedded shape: {embedded.shape}")
print("   ✓ Takens embedding exitoso")

# 4. Test: Homología persistente
print("\n4. Probando homología persistente con Ripser...")
# Usar submuestreo para acelerar
subsample_indices = np.random.choice(embedded.shape[0], size=min(300, embedded.shape[0]), replace=False)
embedded_subsample = embedded[subsample_indices, :]

result = ripser(embedded_subsample, maxdim=2, thresh=2.0)
diagrams = result['dgms']
print(f"   H0 features: {len(diagrams[0])}")
print(f"   H1 features: {len(diagrams[1])}")
print(f"   H2 features: {len(diagrams[2])}")
print("   ✓ Homología persistente exitosa")

# 5. Test: Extracción de características TDA
print("\n5. Probando extracción de características...")
def extract_tda_features(diagram, dim=1):
    """Extraer características de un diagrama de persistencia"""
    if len(diagram) == 0:
        return {
            'n_features': 0,
            'max_persistence': 0,
            'mean_persistence': 0,
            'sum_persistence': 0
        }

    # Remover punto infinito
    diagram_finite = diagram[diagram[:, 1] < np.inf]
    if len(diagram_finite) == 0:
        return {
            'n_features': 0,
            'max_persistence': 0,
            'mean_persistence': 0,
            'sum_persistence': 0
        }

    lifetimes = diagram_finite[:, 1] - diagram_finite[:, 0]

    return {
        'n_features': len(diagram_finite),
        'max_persistence': np.max(lifetimes) if len(lifetimes) > 0 else 0,
        'mean_persistence': np.mean(lifetimes) if len(lifetimes) > 0 else 0,
        'sum_persistence': np.sum(lifetimes)
    }

features_h1 = extract_tda_features(diagrams[1], dim=1)
print(f"   H1 features: {features_h1}")
print("   ✓ Extracción de características exitosa")

# 6. Test: Machine Learning pipeline completo
print("\n6. Probando pipeline completo de ML...")
print("   Generando dataset de entrenamiento...")
n_samples_per_class = 20
X_features = []
y_labels = []

# Generar muestras para ambas clases
for i in range(n_samples_per_class):
    # Interictal
    eeg_inter = generate_realistic_eeg_segment(duration=5, state='interictal')
    eeg_inter = preprocess_eeg(eeg_inter)
    signal_inter = eeg_inter[0, :]  # Canal 0
    embedded_inter = takens_embedding(signal_inter, delay=10, dimension=3)

    # Submuestreo
    subsample_inter = embedded_inter[::5][:300]
    result_inter = ripser(subsample_inter, maxdim=1, thresh=2.0)
    features_inter = extract_tda_features(result_inter['dgms'][1])

    X_features.append([
        features_inter['n_features'],
        features_inter['max_persistence'],
        features_inter['mean_persistence'],
        features_inter['sum_persistence']
    ])
    y_labels.append(0)

    # Ictal
    eeg_ictal = generate_realistic_eeg_segment(duration=5, state='ictal')
    eeg_ictal = preprocess_eeg(eeg_ictal)
    signal_ictal = eeg_ictal[0, :]
    embedded_ictal = takens_embedding(signal_ictal, delay=10, dimension=3)

    subsample_ictal = embedded_ictal[::5][:300]
    result_ictal = ripser(subsample_ictal, maxdim=1, thresh=2.0)
    features_ictal = extract_tda_features(result_ictal['dgms'][1])

    X_features.append([
        features_ictal['n_features'],
        features_ictal['max_persistence'],
        features_ictal['mean_persistence'],
        features_ictal['sum_persistence']
    ])
    y_labels.append(1)

X = np.array(X_features)
y = np.array(y_labels)
print(f"   Dataset shape: {X.shape}")
print(f"   Labels shape: {y.shape}")

# Split y entrenamiento
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print(f"   Train: {X_train.shape}, Test: {X_test.shape}")

# Entrenar modelo
print("   Entrenando Random Forest...")
clf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
clf.fit(X_train, y_train)

# Evaluar
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"   Accuracy: {accuracy:.2%}")
print("\n   Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Interictal', 'Ictal']))
print("   ✓ Pipeline de ML exitoso")

# Resumen final
print("\n" + "=" * 60)
print("RESUMEN DE PRUEBAS")
print("=" * 60)
print("✓ Generación de datos EEG")
print("✓ Preprocesamiento de señales")
print("✓ Takens embedding")
print("✓ Homología persistente (Ripser)")
print("✓ Extracción de características TDA")
print("✓ Pipeline completo de Machine Learning")
print("\n✅ TODAS LAS PRUEBAS PASARON EXITOSAMENTE")
print("=" * 60)
