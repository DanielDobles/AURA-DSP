# AURA-DSP (AARS) 🎧⚡

**Autonomous Audio Restoration Swarm — Powered by AMD ROCm**

AURA-DSP is a mission-critical autonomous audio restoration system designed to transform low-quality recordings into high-fidelity masterpieces with zero human intervention. Utilizing a swarm of coordinated intelligent agents and advanced DSP processing, AURA-DSP redefines speed and precision in sound engineering.

---

## 🚀 The Problem and Our Solution

### The Pain Point
Traditional audio restoration is a manual, slow, and expensive process. A professional sound engineer can take **dozens of hours** to clean, upsample, rebalance, and master a single track using manual tools. In mission-critical environments, this bottleneck is unacceptable.

### Our Solution: AURA-DSP
AURA-DSP automates this entire cycle through a **Circular Collaborative Swarm**. It is not just a script; it is a virtual team of 5 specialists (AI) that analyze the audio's spectral signature, retrieve heuristics from past runs, and execute a surgical-precision DSP toolchain.

---

## ⚡ The AMD Advantage: From Hours to Minutes

The heart of AURA-DSP beats thanks to the **AMD ROCm** architecture. Leveraging the massive parallelism of our AMD cards and PyTorch kernel optimization for ROCm, we have achieved what seemed impossible:

> **What previously took dozens of hours on a conventional workstation, AURA-DSP resolves in just 30 minutes.**

This computing power allows us to simultaneously run massive language models (like Qwen-32B via vLLM) and neural source separation algorithms (Demucs), ensuring the "Wow Factor" is reached in a fraction of traditional time.

---

## 🧠 Architecture: The Collaborative Swarm (OODA Loop)

AURA-DSP is not a linear pipeline; it is a **Mission-Critical Agent Swarm** coordinated via an **OODA** cycle (Observe, Orient, Decide, Act). Each agent operates autonomously over specific DSP tools optimized for **AMD ROCm**.

### Swarm Specialists:

1.  **Chief Sound Engineer (Strategist):**
    *   **Mission:** Orchestration and delegation. Analyzes the spectral fingerprint and consults the **Swarm Memory** to design the optimal restoration strategy.
    *   **AI:** Central coordinator based on CrewAI.

2.  **Physics-Mathematics Engineer:**
    *   **Mission:** Sample-level precision. Executes 96kHz upsampling (Soxr VHQ) and transient preservation via Hilbert transforms, using GPU-accelerated PyTorch kernels.

3.  **Psychoacoustics Expert:**
    *   **Mission:** Emotional impact and clarity. Applies asymmetric saturation algorithms (tube warmth) and dynamic Mid-Side (M/S) stereo expansion.

4.  **Sound Engineer (Neural):**
    *   **Mission:** Intelligent rebalancing. Uses **Demucs (HTDemucs)** models to separate stems, isolate vocals, and reconstruct the rhythmic structure with phase coherence.

5.  **The Listener (QC Auditor):**
    *   **Mission:** The quality guardian. Quantitatively compares the original vs. processed audio, measuring **SNR**, bandwidth, and crest factor deltas. Holds veto power to force retries if the result is not perfect.

---

## 🛠 Installation and Usage

### Requirements
*   AMD Hardware with **ROCm 7.x+** support.
*   Docker & Docker Compose.

### Quick Deployment
To prepare the server and deploy the complete infrastructure (including vLLM):

```bash
bash infra/deploy_all.sh
```

### Audio Processing
Place your files in `/data/input` and run the orchestrator:

```bash
python pipeline/main.py --input /data/input --output /data/output --purge
```

---

## 🛠 Next Steps (Beta v0.9-beta)
While current results are disruptive, we are still working on:
*   **DSP Process Enhancement:** Refining Hilbert transform-based transient preservation algorithms.
*   **Audio Engineering:** Reducing artifacts in high-frequency reconstruction (Super-Resolution).
*   **Memory Optimization:** Reducing VRAM footprint to allow for larger swarms.

---

## 🙏 Acknowledgments
A special thank you to **AMD** for providing us with the hardware and tools necessary to push the limits of what is possible in digital signal processing and artificial intelligence. Without your ROCm ecosystem, this level of performance would not be possible.

---
*AURA-DSP: Audio Restoration at the Speed of Light.*
