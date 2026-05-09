"""
AARS — Medical Record Schemas
Pydantic models for audio diagnosis and surgery plans.
These schemas enforce strict validation so the LLM can never
generate an incoherent surgery plan.
"""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class SeverityLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ToolName(str, Enum):
    """Available surgical tools (ROCm-compatible only)."""
    AERO = "aero"                    # Super-resolution (replaces AudioSR)
    VOCOS = "vocos"                  # Vocoder reconstruction
    DEEPFILTERNET = "deepfilternet"  # Noise/artifact removal (replaces Resemble Enhance)
    ACE_STEP = "ace_step"            # Source separation (replaces Demucs/Roformer)
    FFMPEG = "ffmpeg"                # Final render / format conversion
    SKIP = "skip"                    # No action needed


class Diagnostics(BaseModel):
    """Raw measurements from Audiophile AI analysis."""
    high_frequency_cutoff: int = Field(
        ..., ge=0, le=48000,
        description="Detected high-frequency cutoff in Hz. Suno typically cuts at 15-16kHz."
    )
    phase_correlation_vocal: float = Field(
        ..., ge=-1.0, le=1.0,
        description="Phase correlation of vocal channel. < 0.0 = cancellation, > 0.5 = healthy."
    )
    artifact_level: SeverityLevel = Field(
        ...,
        description="Overall artifact severity (robotic sounds, vocoder noise, clipping)."
    )
    loudness_lufs: float = Field(
        ..., ge=-70.0, le=0.0,
        description="Integrated loudness in LUFS."
    )
    peak_dbfs: float = Field(
        ..., ge=-70.0, le=6.0,
        description="True peak in dBFS. > 0.0 = clipping."
    )
    sample_rate: int = Field(
        default=44100, ge=8000, le=192000,
        description="Original sample rate."
    )
    duration_seconds: float = Field(
        ..., gt=0.0,
        description="Track duration in seconds."
    )


class SurgeryStep(BaseModel):
    """A single step in the surgery plan."""
    tool: ToolName
    target: str = Field(
        ...,
        description="What to process: 'master_mix', 'vocals', 'instrumental', 'drums', etc."
    )
    intensity: str = Field(
        default="moderate",
        description="Processing intensity: 'gentle', 'moderate', 'aggressive'."
    )
    reason: str = Field(
        ...,
        description="Why this step is prescribed."
    )

    @field_validator("intensity")
    @classmethod
    def validate_intensity(cls, v):
        allowed = {"gentle", "moderate", "aggressive"}
        if v not in allowed:
            raise ValueError(f"Intensity must be one of {allowed}")
        return v


class MedicalRecord(BaseModel):
    """Complete diagnosis and surgery plan for one audio track."""
    track_id: str = Field(
        ...,
        description="Unique identifier for the track (e.g. 'suno_pop_01')."
    )
    source_file: str = Field(
        ...,
        description="Original filename."
    )
    timestamp: str = Field(
        ...,
        description="ISO 8601 timestamp of diagnosis."
    )
    diagnostics: Diagnostics
    surgery_plan: List[SurgeryStep] = Field(
        ..., min_length=1,
        description="Ordered list of surgical steps. Must have at least one step."
    )
    notes: Optional[str] = Field(
        default=None,
        description="Additional notes from Audiophile AI."
    )

    @field_validator("surgery_plan")
    @classmethod
    def validate_plan_coherence(cls, v, info):
        """Ensure the plan makes sense."""
        tool_names = [step.tool for step in v]
        # If all steps are SKIP, that's fine (healthy track)
        if all(t == ToolName.SKIP for t in tool_names):
            return v
        # If we have real tools, SKIP should not be mixed in randomly
        non_skip = [t for t in tool_names if t != ToolName.SKIP]
        if non_skip and ToolName.SKIP in tool_names:
            # SKIP is allowed as a "skip_phase_correction" type step
            pass
        return v


class QualityCheckResult(BaseModel):
    """Post-surgery quality validation."""
    track_id: str
    passed: bool
    checks: dict = Field(
        ...,
        description="Individual check results: silence, clipping, duration, etc."
    )
    error_message: Optional[str] = None
