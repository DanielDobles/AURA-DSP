"""
AARS — Autonomous Crew (v3: Circular Collaborative Swarm)

Architecture: 5-Agent Collaborative Circular Loop
  1. Ingeniero de Sonido Jefe (Master SWE, DSP Expert) - Orchestrates and delegates.
  2. Ingeniero de Sonido - Executes EQ, neural rebalance, max loudness.
  3. Experto en Psicoacústica y Audiófilo - Enhances warmth, clarity, width.
  4. Ingeniero Físico-Matemático - Applies strict mathematical DSP (upsampling, transients).
  5. El Oyente - Final quality reviewer.

Communication flow: All agents communicate circularly via delegation, orchestrated by the Chief.
"""

import os
import json
from crewai import Agent, Crew, Process, Task, LLM
from tools.surgeon_tools import (
    AudiophileUpsamplerTool,
    TransientPreservationTool,
    HarmonicExciterTool,
    PsychoacousticExciterTool,
    StereoWidthTool,
    MasterMaximizerTool,
    FFmpegProMasteringTool,
    QualityComparisonTool,
    TonalBalanceStabilizerTool,
    NeuralMasterRebalanceTool
)

# ── LLM Configuration ─────────────────────────────────────

llm_brain = LLM(
    model="openai/Qwen/Qwen3-32B",
    base_url=os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"),
    api_key="none",
    extra_body={"chat_template_kwargs": {"enable_thinking": False}}
)


class AURACrew:
    """Builds and orchestrates the 5-agent collaborative restoration swarm."""

    def __init__(self):
        # Instantiate individual tools for specialized agents
        self._tools_upsampler = AudiophileUpsamplerTool()
        self._tools_transient = TransientPreservationTool()
        self._tools_exciter = HarmonicExciterTool()
        self._tools_psycho = PsychoacousticExciterTool()
        self._tools_wide = StereoWidthTool()
        self._tools_maxi = MasterMaximizerTool()
        self._tools_master = FFmpegProMasteringTool()
        self._tools_qc = QualityComparisonTool()
        self._tools_tonal = TonalBalanceStabilizerTool()
        self._tools_neural = NeuralMasterRebalanceTool()

    def build_crew(self):
        """Constructs the 5-agent circular collaborative crew."""

        # ── 1. Ingeniero de Sonido Jefe / Orchestrator ──
        chief_engineer = Agent(
            role="Ingeniero de Sonido Jefe y Experto en DSP",
            goal=(
                "Orquestar el proceso circular de restauración de audio. Analizas el diagnóstico espectral "
                "y coordinas al equipo delegando la ejecución de herramientas a los especialistas adecuados."
            ),
            backstory=(
                "Eres un Ingeniero de Sonido Jefe con un máster en Ingeniería de Software y Experto en DSP. "
                "Crees en un flujo de trabajo circular y no piramidal: todos los agentes se ayudan mutuamente según sus campos. "
                "Tú orquestas al equipo (Ingeniero de Sonido, Experto en Psicoacústica, Físico-Matemático, y El Oyente). "
                "Diseñas la cadena de herramientas basada en los datos espectrales y delegas cada paso al especialista correspondiente."
            ),
            llm=llm_brain,
            verbose=True,
            allow_delegation=True,
            max_iter=20
        )

        # ── 2. Ingeniero de Sonido ──
        sound_engineer = Agent(
            role="Ingeniero de Sonido",
            goal="Ejecutar herramientas de mezcla y masterización clásicas.",
            backstory=(
                "Eres un Ingeniero de Sonido profesional. Trabajas junto al equipo aplicando tus herramientas especializadas: "
                "ecualización de balance tonal (spectral_tonal_balance_stabilizer), rebalanceo de stems neuronales (neural_master_rebalance), "
                "y maximización final de volumen (dynamic_boost_maximizer, ffmpeg_pro_master)."
            ),
            tools=[self._tools_tonal, self._tools_neural, self._tools_maxi, self._tools_master],
            llm=llm_brain,
            verbose=True,
            allow_delegation=True
        )

        # ── 3. Experto en Psicoacústica y Audiófilo ──
        psychoacoustics_expert = Agent(
            role="Experto en Psicoacústica y Audiófilo",
            goal="Mejorar la percepción espacial, claridad armónica y calidez del audio.",
            backstory=(
                "Eres un Ingeniero de Sonido con máster en psicoacústica y experto audiófilo. "
                "Te encargas de procesar el sonido para el oído humano, otorgando mayor claridad, calidez armónica y una imagen estéreo expansiva. "
                "Utilizas excitadores psicoacústicos (psychoacoustic_clarity_exciter, harmonic_spectral_exciter) y expansores estéreo (stereo_spatial_widener)."
            ),
            tools=[self._tools_exciter, self._tools_psycho, self._tools_wide],
            llm=llm_brain,
            verbose=True,
            allow_delegation=True
        )

        # ── 4. Ingeniero Físico-Matemático ──
        physics_math_engineer = Agent(
            role="Ingeniero de Sonido Físico-Matemático",
            goal="Aplicar transformaciones matemáticas estrictas y físicas a nivel de muestra.",
            backstory=(
                "Eres un Ingeniero de Sonido con máster en Física y Matemáticas. "
                "Tu trabajo es la precisión absoluta: interpolación matemática para upsampling evitando aliasing (soxr_vhq_upsampler) "
                "y cálculo de envolventes de Hilbert para la preservación de transitorios (transient_preservation_dsp)."
            ),
            tools=[self._tools_upsampler, self._tools_transient],
            llm=llm_brain,
            verbose=True,
            allow_delegation=True
        )

        # ── 5. El Oyente ──
        listener_reviewer = Agent(
            role="El Oyente y Revisor Final",
            goal="Auditar el resultado final comparándolo con el original y comunicarse con el equipo para validar el éxito.",
            backstory=(
                "Eres 'El Oyente'. Tienes el oído final. Tu trabajo es ejecutar la herramienta de control de calidad (quality_comparison_auditor) "
                "para medir cuantitativa y cualitativamente si el audio mejoró. Te comunicas con el equipo dando tu veredicto "
                "y feedback detallado para saber si la restauración fue exitosa."
            ),
            tools=[self._tools_qc],
            llm=llm_brain,
            verbose=True,
            allow_delegation=True
        )

        # ── Task: Circular Collaborative Restoration ──
        collaboration_task = Task(
            description=(
                "Inicia el proceso circular de restauración para el archivo: {audio_path} (Filename: {filename})\n"
                "Diagnóstico Espectral: {spectral_json}\n"
                "Memoria del Swarm: {memory_briefing}\n"
                "Objetivo de la Misión: {mission_goal}\n\n"
                "INSTRUCCIONES PARA EL INGENIERO JEFE (TÚ):\n"
                "1. Analiza el diagnóstico espectral. Decide qué herramientas se necesitan.\n"
                "2. DELEGA a tus compañeros (Ingeniero de Sonido, Experto en Psicoacústica, Ingeniero Físico-Matemático) "
                "la ejecución secuencial de sus herramientas. Dales instrucciones claras sobre qué input_path usar "
                "y pídeles que usen output_paths intermedios (ej. /data/intermediate/{filename}_paso1.wav).\n"
                "3. Asegúrate de que el último experto en procesar el audio guarde el resultado final EXACTAMENTE en: "
                "/data/output/{filename}_restored.wav\n"
                "4. Una vez generado el archivo final, DELEGA a 'El Oyente y Revisor Final' para que ejecute el control de calidad "
                "comparando el {audio_path} original con el /data/output/{filename}_restored.wav.\n"
                "5. Devuelve el reporte JSON de 'El Oyente' como el resultado final de esta tarea."
            ),
            expected_output=(
                "El veredicto JSON de El Oyente con las métricas de calidad y el estado qc_passed, que confirme que todos colaboraron."
            ),
            agent=chief_engineer
        )

        # ── Assemble Circular Crew ──
        return Crew(
            agents=[
                chief_engineer, 
                sound_engineer, 
                psychoacoustics_expert, 
                physics_math_engineer, 
                listener_reviewer
            ],
            tasks=[collaboration_task],
            process=Process.sequential,  # We use sequential for the main task, but it operates circularly via allow_delegation
            verbose=True
        )
