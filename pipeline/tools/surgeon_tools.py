"""
AARS Surgeon Tools — CrewAI tool wrappers for audio restoration models.
Each tool runs as a subprocess inside the MI300X container.
"""
import subprocess
import os
from pathlib import Path
from crewai.tools import tool


@tool("aero_super_resolution")
def aero_super_resolution(input_path: str, output_path: str) -> str:
    """Upscale audio frequency content using AERO spectral super-resolution.
    Use when the spectral analysis shows cutoff below 16kHz (Suno ceiling).
    Args:
        input_path: Path to the input WAV file.
        output_path: Path where the enhanced WAV will be saved.
    Returns:
        Status message with output path.
    """
    try:
        result = subprocess.run(
            ["python", "-m", "aero.enhance",
             "--input", input_path,
             "--output", output_path,
             "--sr", "44100"],
            capture_output=True, text=True, timeout=900
        )
        if result.returncode == 0:
            return f"SUCCESS: Super-resolved audio saved to {output_path}"
        error_msg = f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        return f"ERROR: AERO failed:\n{error_msg[-5000:]}"
    except Exception as e:
        return f"ERROR: {str(e)}"


@tool("deepfilter_denoise")
def deepfilter_denoise(input_path: str, output_path: str) -> str:
    """Remove noise and compression artifacts using DeepFilterNet.
    Use when noise_floor_db is above -50dB or the audiophile detects hiss.
    Args:
        input_path: Path to the input audio file.
        output_path: Path where the cleaned audio will be saved.
    Returns:
        Status message with output path.
    """
    try:
        result = subprocess.run(
            ["deepFilter", input_path,
             "--output-dir", str(Path(output_path).parent)],
            capture_output=True, text=True, timeout=600
        )
        if result.returncode == 0:
            return f"SUCCESS: Denoised audio saved to {output_path}"
        error_msg = f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        return f"ERROR: DeepFilter failed:\n{error_msg[-5000:]}"
    except Exception as e:
        return f"ERROR: {str(e)}"


@tool("ace_step_separation")
def ace_step_separation(input_path: str, output_dir: str) -> str:
    """Separate audio into stems (vocals, drums, bass, other) using ACE-Step.
    Use when the architect decides to process vocals and instruments separately.
    Args:
        input_path: Path to the input audio file.
        output_dir: Directory where stems will be saved.
    Returns:
        Status message with list of stem files.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        result = subprocess.run(
            ["python", "-m", "ace_step.separate",
             "--input", input_path,
             "--output-dir", output_dir],
            capture_output=True, text=True, timeout=1200
        )
        if result.returncode == 0:
            stems = list(Path(output_dir).glob("*.wav"))
            return f"SUCCESS: Separated into {len(stems)} stems: {[s.name for s in stems]}"
        return f"ERROR: ACE-Step failed: {result.stderr[:300]}"
    except Exception as e:
        return f"ERROR: {str(e)}"


@tool("ffmpeg_render")
def ffmpeg_render(input_path: str, output_path: str, sample_rate: int = 44100, bit_depth: int = 24) -> str:
    """Final render/conversion using FFmpeg. Ensures output quality standards.
    Args:
        input_path: Path to the processed audio.
        output_path: Final output path (WAV/FLAC/MP3).
        sample_rate: Target sample rate (default 44100).
        bit_depth: Target bit depth (default 24).
    Returns:
        Status message with final file path.
    """
    codec_map = {
        ".wav": ["-c:a", "pcm_s24le"],
        ".flac": ["-c:a", "flac"],
        ".mp3": ["-c:a", "libmp3lame", "-q:a", "0"],
    }
    ext = Path(output_path).suffix.lower()
    codec_args = codec_map.get(ext, ["-c:a", "pcm_s24le"])

    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", input_path,
             "-ar", str(sample_rate)] + codec_args + [output_path],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode == 0:
            size_mb = os.path.getsize(output_path) / (1024 * 1024)
            return f"SUCCESS: Rendered {output_path} ({size_mb:.1f} MB)"
        return f"ERROR: FFmpeg failed: {result.stderr[:300]}"
    except Exception as e:
        return f"ERROR: {str(e)}"


@tool("scipy_emergency_denoise")
def scipy_emergency_denoise(input_path: str, output_path: str) -> str:
    """Robust CPU-based denoising using SciPy. 
    USE THIS AS A FALLBACK if specialized tools (AERO, DeepFilter) fail.
    It performs basic noise reduction and spectral gating.
    Args:
        input_path: Path to the input audio file.
        output_path: Path where the processed file will be saved.
    """
    try:
        import numpy as np
        from scipy.io import wavfile
        from scipy import signal

        rate, data = wavfile.read(input_path)
        if data.dtype != np.float32:
            data = data.astype(np.float32) / 32768.0

        # Basic spectral gate (very rough emergency denoise)
        f, t, Zxx = signal.stft(data, rate, nperseg=1024)
        abs_Zxx = np.abs(Zxx)
        mask = abs_Zxx > (np.mean(abs_Zxx) * 0.5)
        Zxx_cleaned = Zxx * mask
        _, data_cleaned = signal.istft(Zxx_cleaned, rate)

        # Normalize
        data_cleaned = data_cleaned / (np.max(np.abs(data_cleaned)) + 1e-6)
        
        wavfile.write(output_path, rate, (data_cleaned * 32767).astype(np.int16))
        return f"SUCCESS: Emergency CPU denoise completed at {output_path}"
    except Exception as e:
        return f"ERROR: Emergency denoise failed: {str(e)}"
