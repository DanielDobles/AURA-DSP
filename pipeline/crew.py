"""
AARS — Autonomous Crew (v3: Progressive Self-Improvement)

Architecture: 3-agent OODA loop
  1. Strategist   — Reads spectral analysis + swarm memory → produces restoration plan
  2. Surgeon      — Executes the plan using DSP tools → produces restored audio
  3. QC Auditor   — Validates output quality → feeds results back to memory

Communication flow:
  Spectral Data + Memory Briefing → Strategist → Surgeon → QC Auditor → Memory
"""

import os
import json
from crewai import Agent, Crew, Process, Task, LLM
from tools.surgeon_tools import (
    AudiophileUpsamplerTool,
    TransientPreservationTool,
    HarmonicExciterTool,
    FFmpegProMasteringTool,
    QualityComparisonTool
)


# ── LLM Configuration ─────────────────────────────────────

llm_brain = LLM(
    model="openai/Qwen/Qwen3-32B",
    base_url=os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
    api_key="none"
)


class AURACrew:
    """Builds and orchestrates the 3-agent restoration swarm."""

    def __init__(self):
        self._tools_upsampler = AudiophileUpsamplerTool()
        self._tools_transient = TransientPreservationTool()
        self._tools_exciter = HarmonicExciterTool()
        self._tools_master = FFmpegProMasteringTool()
        self._tools_qc = QualityComparisonTool()

    def build_crew(self):
        """Constructs the 3-agent crew with inter-task context flow."""

        # ── Agent 1: Strategist ────────────────────────────
        strategist = Agent(
            role="Restoration Strategist",
            goal=(
                "Analyze the spectral diagnosis and swarm memory briefing to determine "
                "the optimal restoration plan. Output a PRECISE ordered list of tool calls."
            ),
            backstory=(
                "You are an audio forensics expert. You read spectral diagnostics and "
                "historical swarm memory to decide which DSP tools to apply, in what order, "
                "and with what expectations. You never execute tools yourself — you only "
                "produce the strategy. Base your decisions on the spectral data provided: "
                "if cutoff_freq_hz < 16000, upsampling is mandatory. If noise_floor_db > -50, "
                "mastering with denoise is critical. Always prescribe the QC comparison at the end."
            ),
            llm=llm_brain,
            verbose=True,
            allow_delegation=False,
            max_iter=5
        )

        # ── Agent 2: Surgeon ───────────────────────────────
        surgeon = Agent(
            role="Master Audio Surgeon",
            goal=(
                "Execute the restoration plan by calling DSP tools in the exact order "
                "prescribed by the Strategist. YOU MUST CALL THE TOOLS — do not simulate."
            ),
            backstory=(
                "You are a precision execution unit. You receive a restoration plan from "
                "the Strategist and execute each step by calling the appropriate tool with "
                "the correct file paths. After each tool call, report the JSON metrics returned. "
                "Chain outputs: each tool's output_path becomes the next tool's input_path."
            ),
            tools=[
                self._tools_upsampler,
                self._tools_transient,
                self._tools_exciter,
                self._tools_master,
            ],
            llm=llm_brain,
            verbose=True,
            allow_delegation=False,
            max_iter=15
        )

        # ── Agent 3: QC Auditor ────────────────────────────
        qc_auditor = Agent(
            role="Quality Control Auditor",
            goal=(
                "Compare the original audio with the final restored output using the "
                "quality_comparison_auditor tool. Report the verdict with full metrics."
            ),
            backstory=(
                "You are the final checkpoint. You use the quality_comparison_auditor tool "
                "to measure if the restoration improved the audio. You MUST call the tool — "
                "your judgment is based on its quantitative output, not opinion. "
                "Pass the ORIGINAL input file as input_path and the FINAL restored file as output_path."
            ),
            tools=[self._tools_qc],
            llm=llm_brain,
            verbose=True,
            allow_delegation=False,
            max_iter=5
        )

        # ── Task 1: Strategic Analysis ─────────────────────
        strategy_task = Task(
            description=(
                "You are given:\n"
                "- Filename: {filename}\n"
                "- Audio path: {audio_path}\n"
                "- Spectral diagnosis: {spectral_json}\n"
                "- Swarm memory briefing: {memory_briefing}\n\n"
                "Produce a numbered restoration plan. For each step specify:\n"
                "  - Tool name (one of: soxr_vhq_upsampler, transient_preservation_dsp, "
                "harmonic_spectral_exciter, ffmpeg_pro_master)\n"
                "  - input_path and output_path\n"
                "  - Expected improvement\n\n"
                "Rules:\n"
                "  1. If sample_rate < 96000, first step MUST be soxr_vhq_upsampler\n"
                "  2. Always chain outputs: step N output → step N+1 input\n"
                "  3. Final output must be at /data/output/{filename}_restored.wav\n"
                "  4. The QC step (quality_comparison_auditor) will be handled by the next agent"
            ),
            expected_output=(
                "A numbered restoration plan with tool names, file paths, and rationale."
            ),
            agent=strategist
        )

        # ── Task 2: Surgical Execution ─────────────────────
        execution_task = Task(
            description=(
                "Execute the restoration plan from the Strategist.\n"
                "For the file: {audio_path} (named: {filename})\n\n"
                "CRITICAL INSTRUCTIONS:\n"
                "1. Call each tool in the prescribed order using the EXACT paths\n"
                "2. After each tool call, parse the JSON response to confirm success\n"
                "3. If a tool returns success=false, STOP and report the error\n"
                "4. Chain: each tool's output_path becomes the next tool's input_path\n"
                "5. The final output MUST be saved at /data/output/{filename}_restored.wav\n\n"
                "Standard restoration chain (if no specific plan from Strategist):\n"
                "  Step 1: soxr_vhq_upsampler → /data/output/{filename}_96k.wav\n"
                "  Step 2: transient_preservation_dsp → /data/output/{filename}_transient.wav\n"
                "  Step 3: harmonic_spectral_exciter → /data/output/{filename}_excited.wav\n"
                "  Step 4: ffmpeg_pro_master → /data/output/{filename}_restored.wav"
            ),
            expected_output=(
                "A report of each tool execution with the JSON metrics from each step."
            ),
            agent=surgeon,
            context=[strategy_task],
            tools=[
                self._tools_upsampler,
                self._tools_transient,
                self._tools_exciter,
                self._tools_master,
            ]
        )

        # ── Task 3: Quality Control ────────────────────────
        qc_task = Task(
            description=(
                "Validate the restoration quality.\n"
                "Call the quality_comparison_auditor tool with:\n"
                "  input_path = {audio_path}  (the ORIGINAL file)\n"
                "  output_path = /data/output/{filename}_restored.wav  (the RESTORED file)\n\n"
                "Report the full QC verdict including all deltas and pass/fail status."
            ),
            expected_output=(
                "The complete QC verdict JSON with qc_passed, deltas, and issues."
            ),
            agent=qc_auditor,
            context=[execution_task],
            tools=[self._tools_qc]
        )

        # ── Assemble Crew ──────────────────────────────────
        return Crew(
            agents=[strategist, surgeon, qc_auditor],
            tasks=[strategy_task, execution_task, qc_task],
            process=Process.sequential,
            verbose=True
        )
