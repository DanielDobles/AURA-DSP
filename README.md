# AURA-DSP (AARS) 🎧⚡

**Autonomous Audio Restoration Swarm — Powered by AMD ROCm**

AURA-DSP es un sistema de restauración de audio autónomo de misión crítica diseñado para transformar grabaciones de baja calidad en piezas de alta fidelidad con cero intervención humana. Utilizando un enjambre (swarm) de agentes inteligentes coordinados y procesado DSP avanzado, AURA-DSP redefine la velocidad y precisión en la ingeniería de sonido.

---

## 🚀 El Problema y Nuestra Solución

### El Punto de Dolor
La restauración de audio tradicional es un proceso artesanal, lento y costoso. Un ingeniero de sonido profesional puede tardar **decenas de horas** en limpiar, upsamplear, rebalancear y masterizar un solo track utilizando herramientas manuales. En entornos de misión crítica, este cuello de botella es inaceptable.

### Nuestra Solución: AURA-DSP
AURA-DSP automatiza este ciclo completo mediante un **Swarm Colaborativo Circular**. No es solo un script; es un equipo virtual de 5 especialistas (IA) que analizan la firma espectral del audio, recuperan heurísticas de ejecuciones pasadas y ejecutan una cadena de herramientas DSP de precisión quirúrgica.

---

## ⚡ La Ventaja AMD: De Horas a Minutos

El corazón de AURA-DSP late gracias a la arquitectura **AMD ROCm**. Gracias al paralelismo masivo de nuestras tarjetas AMD y la optimización de los kernels de PyTorch para ROCm, hemos logrado lo que parecía imposible:

> **Lo que antes tomaba decenas de horas en una estación de trabajo convencional, AURA-DSP lo resuelve en solo 30 minutos.**

Esta capacidad de cómputo nos permite ejecutar modelos de lenguaje masivos (como Qwen-32B vía vLLM) y algoritmos de separación de fuentes neuronales (Demucs) de forma simultánea, garantizando que el "Wow Factor" se alcance en una fracción del tiempo tradicional.

---

## 🧠 Arquitectura del Sistema (OODA Loop)

El sistema opera bajo un flujo de mejora continua:

1.  **Análisis Espectral:** Escaneo profundo de la firma acústica, ruido de fondo y transitorios.
2.  **Swarm Memory:** Consulta de éxitos previos para ajustar la estrategia.
3.  **Ejecución del Enjambre (Crew):**
    *   **Ingeniero Jefe:** Orquestador y estratega.
    *   **Ingeniero de Sonido:** Especialista en EQ y masterización neuronal.
    *   **Experto en Psicoacústica:** Calidez, claridad y expansión estéreo.
    *   **Ingeniero Físico-Matemático:** Upsampling VHQ y preservación de transitorios.
    *   **El Oyente:** Auditor de control de calidad final.
4.  **Control de Calidad (QC):** Si el veredicto falla, el sistema ajusta parámetros y reintenta automáticamente.

---

## 🛠 Instalación y Uso

### Requisitos
*   Hardware AMD con soporte **ROCm 7.x+**.
*   Docker & Docker Compose.

### Despliegue Rápido
Para preparar el servidor y desplegar la infraestructura completa (incluyendo vLLM):

```bash
bash infra/deploy_all.sh
```

### Procesamiento de Audio
Coloca tus archivos en `/data/input` y ejecuta el orquestador:

```bash
python pipeline/main.py --input /data/input --output /data/output --purge
```

---

## 🛠 Próximos Pasos (Beta v0.9-beta)
Aunque los resultados actuales son disruptivos, seguimos trabajando en:
*   **Mejora de Procesos DSP:** Refinamiento de los algoritmos de preservación de transitorios por transformada de Hilbert.
*   **Ingeniería de Audio:** Reducción de artefactos en la reconstrucción de frecuencias altas (Super-Resolution).
*   **Optimización de Memoria:** Reducción del footprint de VRAM para permitir swarms más grandes.

---

## 🙏 Agradecimientos
Un agradecimiento especial a **AMD** por proporcionarnos el hardware y las herramientas necesarias para empujar los límites de lo que es posible en el procesamiento digital de señales y la inteligencia artificial. Sin su ecosistema ROCm, este nivel de rendimiento no sería posible.

---
*AURA-DSP: Audio Restoration at the Speed of Light.*
