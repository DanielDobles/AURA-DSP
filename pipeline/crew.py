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
    PsychoacousticExciterTool,
    StereoWidthTool,
    MasterMaximizerTool,
    FFmpegProMasteringTool,
    QualityComparisonTool,
    TonalBalanceStabilizerTool,
    NeuralMasterRebalanceTool,
    SOTAMasteringChainTool
)


# ── LLM Configuration ─────────────────────────────────────

llm_brain = LLM(
    model="openai/Qwen/Qwen3-32B",
    base_url=os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"),
    api_key="none",
    extra_body={"chat_template_kwargs": {"enable_thinking": False}}
)


class AURACrew:
    """Builds and orchestrates the 3-agent restoration swarm."""

    def __init__(self):
        self._tools_upsampler = AudiophileUpsamplerTool()
        self._tools_transient = TransientPreservationTool()
        self._tools_exciter = HarmonicExciterTool()
        self._tools_psycho = PsychoacousticExciterTool()
        self._tools_wide = StereoWidthTool()
        self._tools_maxi = MasterMaximizerTool()
        self._tools_master = FFmpegProMasteringTool()
        self._tools_qc = QualityComparisonTool()
        self._tools_chain = SOTAMasteringChainTool()

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
                "You are an elite DSP mastering engineer modeled after the core architecture of iZotope Ozone 11. "
                "You read spectral diagnostics and historical swarm memory to decide which DSP tools to apply. "
                "You use a SOTA 'Master Assistant' paradigm. "
                "To achieve an industry-standard, commercial master (Consumer Mode), your pipeline must prioritize: "
                "1. spectral_tonal_balance_stabilizer FIRST to fix muddy/harsh macroscopic frequency response. "
                "2. neural_master_rebalance to perfectly isolate and widen the melody while keeping vocals pristine and mono-bassing the rhythm. "
                "3. psychoacoustic_clarity_exciter for asymmetric tube-like saturation (warmth and bite). "
                "4. transient_preservation_dsp for punch recovery via fast/slow envelope analysis. "
                "5. dynamic_boost_maximizer for GPU Tensor soft-knee maximization to commercial LUFS. "
                "Base your decisions on the spectral data provided: if cutoff_freq_hz < 16000, upsampling is mandatory."
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
                "Execute ALL steps of the restoration plan sequentially by calling each DSP tool "
                "one at a time in order. You MUST call each tool — never simulate or skip. "
                "NEVER call quality_comparison_auditor — that tool belongs to a different agent."
            ),
            backstory=(
                "You are a deterministic execution engine. You receive a numbered restoration plan and "
                "you execute EACH step by calling the correct tool with the exact paths specified. "
                "After each tool call, parse the JSON response. If success=true, proceed to the next step "
                "using the output_path as the next input_path. If success=false, STOP and report the error. "
                "You must call ALL tools in the plan. Do NOT skip any step. Do NOT use quality_comparison_auditor. "
                "Your job is ONLY to execute DSP tools in sequence and report their results."
            ),
            tools=[self._tools_chain],
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
                "- Swarm memory briefing: {memory_briefing}\n"
                "- MISSION GOAL: {mission_goal}\n\n"
                "Produce a restoration plan. For the step specify:\n"
                "  - Tool name MUST BE: sota_mastering_chain\n"
                "  - input_path and output_path\n"
                "  - Expected improvement\n\n"
                "Rules:\n"
                "  1. Final output must be at /data/output/{filename}_restored.wav\n"
                "  2. The QC step (quality_comparison_auditor) will be handled by the next agent"
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
                "MANDATORY PROTOCOL:\n"
                "1. Call sota_mastering_chain with input_path={audio_path} output_path=/data/output/{filename}_restored.wav\n"
                "2. When the tool completes, output its JSON result.\n"
                "NEVER call quality_comparison_auditor."
            ),
            expected_output=(
                "The JSON result of the sota_mastering_chain tool execution."
            ),
            agent=surgeon,
            context=[strategy_task],
            tools=[self._tools_chain]
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
