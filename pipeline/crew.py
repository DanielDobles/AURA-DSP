from crewai import Agent, Crew, Process, Task, LLM
from tools.surgeon_tools import (
    aero_super_resolution,
    deepfilter_denoise,
    ace_step_separation,
    ffmpeg_render,
)
import os

# Configuración de los modelos vía vLLM (Compatible con OpenAI API)
# Qwen3-32B para razonamiento complejo
llm_brain = LLM(
    model="openai/Qwen/Qwen3-32B",
    base_url=os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
    api_key="none"
)

# Qwen2-Audio para análisis multimodal (escucha)
llm_audio = LLM(
    model="openai/Qwen/Qwen2-Audio-7B-Instruct",
    base_url=os.getenv("VLLM_AUDIO_URL", "http://localhost:8001/v1"),
    api_key="none"
)

class AURACrew:
    """Orchestrates the Audio Restoration Swarm."""
    
    SURGEON_TOOLS = [
        aero_super_resolution,
        deepfilter_denoise,
        ace_step_separation,
        ffmpeg_render,
    ]

    def __init__(self, data_path="/data"):
        self.data_path = data_path

    def architect(self) -> Agent:
        return Agent(
            role='Lead Audio Architect',
            goal='Design a restoration plan for {audio_path}. Choose the best tools from the available toolset.',
            backstory="""You design the blueprint for restoration. You prioritize quality but are realistic about what tools can do.
            Rules:
            - If cutoff_freq < 16000Hz → MUST use aero_super_resolution first
            - If noise_floor_db > -50dB → MUST use deepfilter_denoise
            - Always finish with ffmpeg_render to standardize output format""",
            llm=llm_brain,
            verbose=True,
            allow_delegation=True
        )

    def audiophile(self) -> Agent:
        return Agent(
            role='Multimodal Audiophile Analyst',
            goal='Analyze the audio file {audio_path} and identify subjective artifacts using spectral data {spectral_data}.',
            backstory="""You are a world-class audio engineer. You look at data and 'hear' the problems: hiss, muffled frequencies, glitches.
            You detect "underwater" sounds typical of Suno/Udio generation and provide a human-like description of what needs fixing.""",
            llm=llm_audio,
            verbose=True
        )

    def surgeon(self) -> Agent:
        return Agent(
            role='DSP Surgeon',
            goal='Execute the restoration plan for {audio_path}. If a tool fails (ERROR), you MUST find an alternative way to process the file (e.g., use a different tool or try a fallback strategy).',
            backstory="""You are the expert executioner of DSP tools. You don't give up. 
            Your environment is an AMD MI300X with ROCm. If a GPU tool fails, you try another enhancer or a CPU fallback.
            You must ALWAYS execute the tool and report the SUCCESS or ERROR message exactly.""",
            llm=llm_brain,
            tools=self.SURGEON_TOOLS,
            verbose=True,
            allow_delegation=False
        )

    def controller(self) -> Agent:
        return Agent(
            role='AARS Mission Controller',
            goal='Ensure the mission for {audio_path} is completed. If the Surgeon reports a technical error, you MUST analyze the log and instruct the Surgeon on an alternative strategy.',
            backstory="""You are the SRE (Site Reliability Engineer) of the swarm. Your job is zero downtime. 
            If the environment is broken for one tool (e.g. ABI mismatch, CUDA error), you pivot to another strategy or tool. 
            You are obsessed with finishing the queue with the best possible result.""",
            llm=llm_brain,
            verbose=True,
            allow_delegation=True
        )

    def build_crew(self) -> Crew:
        return Crew(
            agents=[self.architect(), self.audiophile(), self.surgeon(), self.controller()],
            process=Process.sequential,
            verbose=True
        )
