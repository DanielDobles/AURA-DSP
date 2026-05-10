<div align="center">
  <a href="https://lablab.ai/ai-hackathons/amd-developer">
    <img src="https://storage.googleapis.com/lablab-static-eu/images/events/amd-developer/amd-developer-cover.jpg" alt="AMD Developer Hackathon" width="800"/>
  </a>
</div>

# AURA-DSP (AARS) 🎧⚡

**Autonomous Audio Restoration Swarm — Powered by AMD ROCm**

AURA-DSP is a mission-critical autonomous audio restoration system designed to transform low-quality recordings into high-fidelity masterpieces with zero human intervention. Utilizing a swarm of coordinated intelligent agents and advanced DSP processing, AURA-DSP redefines speed and precision in sound engineering.

---

## 🏎️ Hardware Infrastructure (AMD DevCloud)

AURA-DSP was architected, and deployed exclusively on the **[AMD DevCloud](https://devcloud.amd.com/)**. The sheer computational power and massive VRAM bandwidth of the MI300X is what makes this multi-agent swarm architecture possible.

| Component | Specification |
|---|---|
| **GPU** | 1x AMD Instinct™ MI300X |
| **VRAM** | 192 GB High-Bandwidth Memory (HBM3) |
| **CPU** | 20 vCPUs |
| **RAM** | 240 GB System Memory |
| **Storage (Boot)** | 720 GB NVMe SSD |
| **Storage (Staging)**| 5 TB NVMe SSD (High-IOPS scratch disk for audio IO) |

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

## 🧠 Architecture: Why a Multi-Agent Swarm?

### The Problem with Linear Pipelines

A naive approach to audio restoration would be a fixed script: upsample → denoise → EQ → export. But this fails in practice because **every audio file is different**. A podcast needs denoising but not stereo widening. A compressed music track needs spectral reconstruction but not noise reduction. A field recording needs both. Hardcoding the order, parameters, or even *which* tools to apply produces inconsistent results — exactly the kind of one-size-fits-all approach a professional sound engineer would never use.

### The Solution: Agentic Decision-Making with CrewAI

AURA-DSP solves this with **[CrewAI](https://www.crewai.com/)**, a multi-agent orchestration framework. Instead of a rigid script, a team of specialized AI agents *reasons* about each file's spectral characteristics and *decides* the optimal restoration chain at runtime.

**Why CrewAI specifically?**
- **Tool delegation:** Each agent has access to *only* the DSP tools relevant to its specialty. The orchestrator delegates tasks to the right specialist, preventing destructive over-processing.
- **Self-hosted LLM:** All reasoning runs on a **locally deployed Qwen3-32B** via vLLM — no API keys, no cloud inference costs, no data leaving your server.
- **Memory across runs:** A persistent `MemoryStore` tracks past restoration outcomes (SNR deltas, QC verdicts), enabling the swarm to learn from its own mistakes over time.

### The OODA Loop

The swarm operates on an **OODA** cycle (Observe → Orient → Decide → Act):

```
┌─────────────────────────────────────────────────────────────────┐
│                         OODA CYCLE                              │
│                                                                 │
│   OBSERVE          ORIENT            DECIDE          ACT        │
│   ┌──────────┐    ┌──────────────┐  ┌───────────┐  ┌────────┐  │
│   │ Spectral  │───▶│ Chief Sound │──▶│ Delegate  │──▶│ Execute│  │
│   │ Diagnosis │    │ Engineer    │   │ to the    │   │ DSP    │  │
│   │ + Memory  │    │ (Strategist)│   │ right     │   │ Tools  │  │
│   │ Briefing  │    │             │   │ specialist│   │ on GPU │  │
│   └──────────┘    └──────────────┘  └───────────┘  └───┬────┘  │
│                                                        │        │
│                        ┌───────────────────────────────┘        │
│                        ▼                                        │
│                   ┌──────────┐    Pass? ──▶ Output              │
│                   │ QC Audit │                                   │
│                   │ (Listener)│    Fail? ──▶ Retry with feedback │
│                   └──────────┘                                   │
└─────────────────────────────────────────────────────────────────┘
```

### The 5 Specialists

| # | Agent | Role | DSP Tools | Why It Exists |
|---|---|---|---|---|
| 1 | **Chief Sound Engineer** | Orchestrator — reads spectral diagnosis + swarm memory, designs the restoration plan, delegates to specialists | *None (delegates only)* | A human lead engineer wouldn't touch EQ knobs; they'd tell the team *what* to do. Same principle. |
| 2 | **Physics-Math Engineer** | Sample-level precision — upsampling and transient recovery | `soxr_vhq_upsampler`, `transient_preservation_dsp` | Mathematical operations (interpolation, Hilbert transforms) require strict signal theory, not creative judgement. |
| 3 | **Psychoacoustics Expert** | Perceptual enhancement — warmth, clarity, spatial width | `harmonic_spectral_exciter`, `psychoacoustic_clarity_exciter`, `stereo_spatial_widener` | These are subjective, ear-oriented processes. Isolating them prevents over-saturation when combined with EQ. |
| 4 | **Sound Engineer** | Classical mixing and mastering — EQ, stem rebalancing, loudness | `spectral_tonal_balance_stabilizer`, `neural_master_rebalance`, `dynamic_boost_maximizer`, `ffmpeg_pro_master` | Handles both traditional (FFmpeg/FFT) and neural (Demucs stem separation) mastering in one agent. |
| 5 | **The Listener (QC Auditor)** | Quality gate — quantitative comparison of original vs. restored | `quality_comparison_auditor` | The final checkpoint. Measures SNR, bandwidth, and crest factor deltas. Can **reject** a restoration if metrics degrade, forcing a retry. |

### GPU-Accelerated DSP Toolchain

All DSP tools are implemented as Python classes extending CrewAI's `BaseTool` interface. Internally, they leverage:

- **PyTorch + torchaudio** on ROCm for GPU-accelerated resampling, limiting, and M/S processing
- **Demucs (HTDemucs)** for neural source separation (vocals, drums, bass, melody) directly on the MI300X
- **SciPy** for precision filter design (Butterworth, FIR, Savitzky-Golay smoothing)
- **FFmpeg** for spectral denoising (`afftdn`), NL-means denoising (`anlmdn`), and EBU R128 loudness normalization

Every tool returns structured JSON with before/after metrics, enabling the QC Auditor to make data-driven pass/fail decisions.

---

## 🛠 Installation and Usage

All interaction with AURA-DSP is performed through the **`hd.py`** command-line hub.
It acts as the single control plane for infrastructure provisioning, model deployment,
audio processing, telemetry, and file transfer — all from your local workstation (Windows, macOS, or Linux).

### Prerequisites

| Requirement | Details |
|---|---|
| **Remote GPU Server** | An AMD cloud instance (MI300X, MI250, or compatible) with **ROCm 7.x+** and **Docker 24+**. Any provider works (AMD Cloud, Lambda, RunPod, etc.). |
| **SSH Access** | Root-level SSH access to the GPU server via key-based authentication. |
| **Python 3.10+** | On your local machine. Only the `rich` package is required locally (`pip install rich`). |
| **Network** | Outbound SSH (port 22) access to the GPU server from your workstation. |

### Configuration

Before first use, copy `.env.example` to `.env` and fill in your server details:

```env
# .env — AARS Connection Configuration
DROPLET_IP=<your-server-ip>          # e.g. 203.0.113.42
SSH_KEY_PATH=~/.ssh/my_gpu_key       # Path to your SSH private key
```

Alternatively, you can edit the configuration block at the top of `hd.py` directly:

```python
SERVER_IP   = "<your-server-ip>"
SERVER_USER = "root"
SSH_KEY     = os.path.expanduser("~/.ssh/my_gpu_key")
```

### Project Structure

```
AURA-DSP/
├── hd.py                  # 🎯 CLI Hub — the only file you run locally
├── infra/                 # Server provisioning & vLLM deployment scripts
│   ├── setup_server.sh    #   OS hardening, Docker, ROCm drivers
│   └── deploy_vllm.sh     #   Sequential Qwen3 + Qwen2-Audio launch
├── pipeline/              # Restoration swarm (runs inside Docker on GPU)
│   ├── main.py            #   Pipeline entrypoint
│   ├── crew.py            #   CrewAI agent definitions (OODA loop)
│   └── tools/
│       └── surgeon_tools.py  # DSP tool implementations (GPU-accelerated)
├── bridge/                # Local ↔ Server file sync daemon
├── output/                # Downloaded restored files land here
└── README.md
```

---

### 🔥 First-Time Setup: The Ignition Sequence

The `ignite` command performs the full deployment in a single invocation:

```bash
python hd.py ignite
```

This executes four phases automatically:

| Phase | Action | Duration |
|:---:|---|---|
| **1** | Local pre-flight checks (SSH key, directory structure) | ~2 s |
| **2** | Server hardening — installs Docker, ROCm drivers, firewall rules | ~30 s |
| **3** | Deploys Qwen3-32B (brain) and Qwen2-Audio-7B (listener) via vLLM | ~3-5 min |
| **4** | Builds the `aars-pipeline` Docker image on the GPU server | ~20 s |

> **Note:** Phase 3 serializes model loading — Qwen3 is fully loaded into VRAM before Qwen2-Audio begins — to prevent ROCm memory contention on the MI300X.

After ignition, the system auto-starts the **Bridge Sync** daemon for continuous file monitoring.

---

### 📖 CLI Reference

#### Core Operations

| Command | Description |
|---|---|
| `python hd.py ignite` | 🔥 Full system deployment (infrastructure + models + pipeline) |
| `python hd.py run` | ▶️  Execute the restoration pipeline on queued audio |
| `python hd.py process [path]` | 🔥 Shortcut for `run --purge --overwrite` (recommended for fresh runs) |
| `python hd.py build` | 🔨 Rebuild only the pipeline container (after code changes) |
| `python hd.py status` | 📊 Live swarm health + mission telemetry dashboard |

#### File Management

| Command | Description |
|---|---|
| `python hd.py upload <file1> [file2 ...]` | ⬆️  Upload WAV files to the server's `/data/input/` |
| `python hd.py fetch [--dest DIR]` | ⬇️  Download all restored files to local `output/` |
| `python hd.py download [--dest DIR]` | ⬇️  Alias for `fetch` |

#### Diagnostics

| Command | Description |
|---|---|
| `python hd.py logs [--lines N]` | 📜 Show vLLM container logs (Qwen3 + Qwen2-Audio) |
| `python hd.py logs --pipeline` | 📜 Show restoration swarm container logs |
| `python hd.py gpu` | 🏎️  Display real-time GPU metrics via `rocm-smi` |
| `python hd.py cost` | 💰 Estimate current session cost based on server uptime |
| `python hd.py shell` | 🖥️  Open an interactive SSH session to the GPU server |

#### Maintenance

| Command | Description |
|---|---|
| `python hd.py clean` | 🧹 Delete all reports and restored files on the server (with confirmation) |
| `python hd.py stop` | ⏹️  Gracefully stop all running containers |
| `python hd.py purge` | ☠️  Force-kill and remove **all** Docker containers on the server |

---

### Detailed Command Usage

#### `run` — Execute Restoration

```bash
# Process files already on the server at /data/input
python hd.py run

# Process from a local folder (auto-uploads via SCP)
python hd.py run --input ~/Music/MyTracks       # macOS / Linux
python hd.py run --input "C:\Music\MyTracks"      # Windows

# Force fresh processing, overwriting any previous results
python hd.py run --purge --overwrite

# Non-interactive mode (skip confirmations)
python hd.py run --input ./my_tracks -y
```

| Flag | Default | Description |
|---|---|---|
| `--input PATH` | `/data/input` | Path to audio files. Local paths are auto-uploaded; remote paths are used directly. |
| `--purge` | `false` | Clear intermediate files before processing for a clean run. |
| `--overwrite` | `false` | Overwrite existing output files and QC reports. |
| `-y, --yes` | `false` | Auto-confirm all interactive prompts. |

#### `process` — Fresh Dynamic Processing (Recommended)

```bash
# Equivalent to: python hd.py run --purge --overwrite
python hd.py process

# With a custom input path
python hd.py process ~/Music/Album
```

#### `upload` — Batch File Upload

```bash
python hd.py upload track_A.wav track_B.wav "My Song.wav"
```

Each file is transferred via SCP with MD5 integrity verification. Failed transfers are reported individually.

#### `fetch` / `download` — Retrieve Results

```bash
# Download to default ./output/ directory
python hd.py fetch

# Download to a custom directory
python hd.py fetch --dest ~/Desktop/Restored
```

Features automatic skip for already-downloaded files and race-condition handling for files being moved by the Bridge daemon.

#### `logs` — Container Diagnostics

```bash
# Last 50 lines of vLLM model logs (default)
python hd.py logs

# Last 200 lines
python hd.py logs --lines 200

# Restoration pipeline container logs
python hd.py logs --pipeline
```

#### `status` — System Health Dashboard

```bash
python hd.py status
```

Displays two panels:
1. **Live Swarm Status** — Real-time health of Qwen3 Brain, Qwen2-Audio Hearing, and AMD MI300X GPU utilization.
2. **Mission Telemetry** — Per-track processing status, remote file sizes, and local sync state.

---

### ⚡ Recommended Workflows

#### Standard Restoration Run

```bash
# 1. Check that the swarm is online
python hd.py status

# 2. Upload your audio files
python hd.py upload track1.wav track2.wav track3.wav

# 3. Execute the restoration pipeline
python hd.py run --purge

# 4. Results are auto-downloaded to ./output/
```

#### After Modifying DSP Tools

```bash
# Rebuild the pipeline container with your changes
python hd.py build

# Run with fresh state
python hd.py process
```

#### Cold Start After Server Reboot

```bash
# Full infrastructure + model deployment
python hd.py ignite

# Verify everything is online
python hd.py status
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
