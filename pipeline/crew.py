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
            goal='Design the perfect restoration pipeline for each track',
            backstory="""You are a world-class mastering engineer with 30 years of experience.
            You interpret technical spectral data and multimodal audio reports to decide 
            exactly which AI models (AERO, DeepFilterNet, ACE-Step) should be applied 
            and in what order to achieve transparent, high-fidelity sound.
            
            Rules:
            - If cutoff_freq < 16000Hz → MUST use aero_super_resolution first
            - If noise_floor_db > -50dB → MUST use deepfilter_denoise
            - If the audiophile detects overlapping instruments → consider ace_step_separation
            - Always finish with ffmpeg_render to standardize output format""",
            llm=llm_brain,
            verbose=True,
            allow_delegation=True
        )

    def audiophile(self) -> Agent:
        return Agent(
            role='Multimodal Audiophile Analyst',
            goal='Listen to the audio and detect subjective artifacts that DSP misses',
            backstory="""You are the "ears" of the system. You use Qwen2-Audio to hear 
            compression artifacts, metallic cymbals, or "underwater" sounds typical 
            of Suno/Udio generation. You provide a human-like description of what 
            needs fixing.
            
            Focus on:
            - Metallic or ringing high frequencies (vocoder artifacts)
            - "Underwater" or muffled quality (bandwidth limitation)
            - Inconsistent stereo image
            - Rhythmic clicking or digital glitches
            - Unnatural vocal timbre""",
            llm=llm_audio,
            verbose=True
        )

    def surgeon(self) -> Agent:
        return Agent(
            role='DSP Surgeon',
            goal='Execute the restoration tools with surgical precision',
            backstory="""You are an expert in command-line audio processing. 
            You take the instructions from the Architect and run the specific 
            ROCm-optimized tools. 
            
            CRITICAL: You must ALWAYS execute the tool for every step of the plan. 
            Do NOT just describe what you will do. RUN the tool, wait for the 
            SUCCESS or ERROR message, and then proceed to the next tool in the plan.
            If a tool fails, report the error exactly.
            
            Your environment is an AMD MI300X with ROCm. Ensure bit-perfect execution.""",
            llm=llm_brain,
            tools=self.SURGEON_TOOLS,
            verbose=True,
            allow_delegation=False
        )

    def build_crew(self) -> Crew:
        return Crew(
            agents=[self.architect(), self.audiophile(), self.surgeon()],
            process=Process.sequential,
            verbose=True
        )
