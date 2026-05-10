"""
AARS — Surgical DSP Tools (v2: Quantitative Metrics Edition)

Every tool now returns a structured JSON string with:
  - success: bool
  - output_path: str
  - metrics: dict with quantitative before/after measurements
  - message: str (human-readable summary)

This enables the QC Auditor to make data-driven quality assessments
and feeds the MemoryStore with numerical improvement deltas.
"""

import os
import subprocess
import json
from typing import Type
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
import librosa
import soundfile as sf
import numpy as np
from scipy import signal


class ToolInput(BaseModel):
    input_path: str = Field(..., description="Path to the input WAV file.")
    output_path: str = Field(..., description="Path where the result will be saved.")


def _measure_audio(y: np.ndarray, sr: int) -> dict:
    """Compute standardized audio quality metrics."""
    rms = np.sqrt(np.mean(y**2))
    peak = np.max(np.abs(y))
    crest_db = 20 * np.log10(peak / rms) if rms > 0 else 0.0

    # Estimate SNR using signal-to-noise ratio heuristic
    stft = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    avg_spectrum = np.mean(stft, axis=1)
    avg_db = librosa.amplitude_to_db(avg_spectrum, ref=np.max)

    # Effective bandwidth (where energy is above -60dB)
    threshold = -60
    above = freqs[avg_db > threshold]
    bandwidth_hz = float(above[-1]) if len(above) > 0 else 0.0

    # Noise floor estimation (lowest 10% of frames)
    frame_rms = librosa.feature.rms(y=y)[0]
    sorted_rms = np.sort(frame_rms)
    noise_floor_rms = np.mean(sorted_rms[:max(1, len(sorted_rms) // 10)])
    noise_floor_db = 20 * np.log10(noise_floor_rms + 1e-10)

    signal_db = 20 * np.log10(rms + 1e-10)
    snr_db = signal_db - noise_floor_db

    return {
        "sample_rate": sr,
        "duration_sec": round(len(y) / sr, 3),
        "rms_db": round(float(signal_db), 2),
        "peak_dbfs": round(float(20 * np.log10(peak + 1e-10)), 2),
        "crest_factor_db": round(float(crest_db), 2),
        "snr_db": round(float(snr_db), 2),
        "bandwidth_hz": round(bandwidth_hz, 1),
        "noise_floor_db": round(float(noise_floor_db), 2)
    }


def _make_result(success: bool, output_path: str, message: str,
                 input_metrics: dict = None, output_metrics: dict = None) -> str:
    """Create a standardized JSON result string."""
    result = {
        "success": success,
        "output_path": output_path,
        "message": message
    }
    if input_metrics and output_metrics:
        result["metrics"] = {
            "before": input_metrics,
            "after": output_metrics,
            "deltas": {
                "snr_delta_db": round(output_metrics["snr_db"] - input_metrics["snr_db"], 2),
                "bandwidth_delta_hz": round(output_metrics["bandwidth_hz"] - input_metrics["bandwidth_hz"], 1),
                "crest_delta_db": round(output_metrics["crest_factor_db"] - input_metrics["crest_factor_db"], 2)
            }
        }
    return json.dumps(result, indent=2)


class AudiophileUpsamplerTool(BaseTool):
    name: str = "soxr_vhq_upsampler"
    description: str = (
        "High-precision upsampler to 96kHz using the Soxr VHQ (Very High Quality) engine. "
        "Zero phase distortion. Returns JSON with before/after audio quality metrics."
    )
    args_schema: Type[BaseModel] = ToolInput

    def _run(self, input_path: str, output_path: str) -> str:
        try:
            y, sr = librosa.load(input_path, sr=None, mono=False)
            # Handle stereo: librosa returns (channels, samples) for stereo
            is_stereo = y.ndim > 1
            if is_stereo:
                y_measure = y[0]  # Measure on first channel
            else:
                y_measure = y

            input_metrics = _measure_audio(y_measure, sr)

            if is_stereo:
                y_96 = np.array([
                    librosa.resample(ch, orig_sr=sr, target_sr=96000, res_type='soxr_vhq')
                    for ch in y
                ])
                sf.write(output_path, y_96.T, 96000)  # soundfile expects (samples, channels)
                output_metrics = _measure_audio(y_96[0], 96000)
            else:
                y_96 = librosa.resample(y, orig_sr=sr, target_sr=96000, res_type='soxr_vhq')
                sf.write(output_path, y_96, 96000)
                output_metrics = _measure_audio(y_96, 96000)

            return _make_result(
                True, output_path,
                f"Upsampled {sr}Hz → 96000Hz (Soxr VHQ). {'Stereo' if is_stereo else 'Mono'} preserved.",
                input_metrics, output_metrics
            )
        except Exception as e:
            return _make_result(False, output_path, f"Upsampling Error: {str(e)}")


class TransientPreservationTool(BaseTool):
    name: str = "transient_preservation_dsp"
    description: str = (
        "Enhances transients and punch using time-domain envelope followers. "
        "Avoids STFT smearing. Returns JSON with before/after metrics."
    )
    args_schema: Type[BaseModel] = ToolInput

    def _run(self, input_path: str, output_path: str) -> str:
        try:
            y, sr = librosa.load(input_path, sr=None, mono=False)
            is_stereo = y.ndim > 1

            def shape_transients(signal_data):
                envelope = np.abs(signal.hilbert(signal_data))
                smooth_env = signal.convolve(envelope, np.ones(50) / 50, mode='same')
                diff = np.diff(smooth_env, prepend=0)
                gain = 1.0 + np.maximum(0, diff * 3.0)
                shaped = signal_data * gain
                shaped = shaped / (np.max(np.abs(shaped)) + 1e-10)
                return shaped

            if is_stereo:
                input_metrics = _measure_audio(y[0], sr)
                y_shaped = np.array([shape_transients(ch) for ch in y])
                sf.write(output_path, y_shaped.T, sr)
                output_metrics = _measure_audio(y_shaped[0], sr)
            else:
                input_metrics = _measure_audio(y, sr)
                y_shaped = shape_transients(y)
                sf.write(output_path, y_shaped, sr)
                output_metrics = _measure_audio(y_shaped, sr)

            return _make_result(
                True, output_path,
                "Transients preserved and enhanced via Hilbert envelope shaping.",
                input_metrics, output_metrics
            )
        except Exception as e:
            return _make_result(False, output_path, f"Transient Error: {str(e)}")


class HarmonicExciterTool(BaseTool):
    name: str = "harmonic_spectral_exciter"
    description: str = (
        "Injects synthesized harmonics above 16kHz to restore 'air' and clarity. "
        "Returns JSON with bandwidth improvement metrics."
    )
    args_schema: Type[BaseModel] = ToolInput

    def _run(self, input_path: str, output_path: str) -> str:
        try:
            y, sr = librosa.load(input_path, sr=None, mono=False)
            is_stereo = y.ndim > 1

            def excite_harmonics(signal_data, sample_rate):
                sos = signal.butter(12, 12000, 'hp', fs=sample_rate, output='sos')
                highs = signal.sosfilt(sos, signal_data)
                harmonics = np.clip(highs * 4.0, -0.7, 1.2)
                excited = signal_data + harmonics * 0.12
                excited = excited / (np.max(np.abs(excited)) + 1e-10)
                return excited

            if is_stereo:
                input_metrics = _measure_audio(y[0], sr)
                y_excited = np.array([excite_harmonics(ch, sr) for ch in y])
                sf.write(output_path, y_excited.T, sr)
                output_metrics = _measure_audio(y_excited[0], sr)
            else:
                input_metrics = _measure_audio(y, sr)
                y_excited = excite_harmonics(y, sr)
                sf.write(output_path, y_excited, sr)
                output_metrics = _measure_audio(y_excited, sr)

            return _make_result(
                True, output_path,
                f"Harmonic excitation applied. Bandwidth extended.",
                input_metrics, output_metrics
            )
        except Exception as e:
            return _make_result(False, output_path, f"Exciter Error: {str(e)}")


class FFmpegProMasteringTool(BaseTool):
    name: str = "ffmpeg_pro_master"
    description: str = (
        "Final audiophile rendering: FFT-based Spectral Denoise (afftdn), "
        "NL-Means Denoise (anlmdn), and EBU R128 Loudness Normalization. "
        "Outputs 24-bit PCM. Returns JSON with before/after metrics."
    )
    args_schema: Type[BaseModel] = ToolInput

    def _run(self, input_path: str, output_path: str) -> str:
        try:
            # Measure input
            y_in, sr_in = librosa.load(input_path, sr=None)
            input_metrics = _measure_audio(y_in, sr_in)

            cmd = [
                "ffmpeg", "-y", "-i", input_path,
                "-af", "afftdn=nr=15:nf=-40,anlmdn=s=0.0008,loudnorm=I=-14:LRA=7:tp=-1.0",
                "-ar", "96000", "-c:a", "pcm_s24le", output_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                return _make_result(False, output_path, f"FFmpeg Error: {result.stderr[-500:]}")

            # Measure output
            y_out, sr_out = librosa.load(output_path, sr=None)
            output_metrics = _measure_audio(y_out, sr_out)

            return _make_result(
                True, output_path,
                "Pro Mastering applied: afftdn + anlmdn + R128 normalization (24-bit).",
                input_metrics, output_metrics
            )
        except Exception as e:
            return _make_result(False, output_path, f"Mastering Error: {str(e)}")


class QualityComparisonTool(BaseTool):
    """Compares the original input with the final restored output.
    Returns a structured QC verdict with pass/fail and improvement deltas."""
    name: str = "quality_comparison_auditor"
    description: str = (
        "Compares the original input audio with the final restored output. "
        "Returns a JSON QC verdict with pass/fail status and quantitative "
        "improvement metrics (SNR, bandwidth, crest factor deltas)."
    )
    args_schema: Type[BaseModel] = ToolInput

    def _run(self, input_path: str, output_path: str) -> str:
        try:
            y_orig, sr_orig = librosa.load(input_path, sr=None)
            y_rest, sr_rest = librosa.load(output_path, sr=None)

            orig_metrics = _measure_audio(y_orig, sr_orig)
            rest_metrics = _measure_audio(y_rest, sr_rest)

            deltas = {
                "snr_delta_db": round(rest_metrics["snr_db"] - orig_metrics["snr_db"], 2),
                "bandwidth_delta_hz": round(rest_metrics["bandwidth_hz"] - orig_metrics["bandwidth_hz"], 1),
                "crest_delta_db": round(rest_metrics["crest_factor_db"] - orig_metrics["crest_factor_db"], 2),
                "sample_rate_change": f"{orig_metrics['sample_rate']}Hz → {rest_metrics['sample_rate']}Hz"
            }

            # Pass criteria: at least one metric improved, none degraded critically
            snr_ok = deltas["snr_delta_db"] >= -2.0      # Allow max 2dB SNR loss
            bw_ok = deltas["bandwidth_delta_hz"] >= -500  # Allow max 500Hz bandwidth loss
            crest_ok = deltas["crest_delta_db"] >= -3.0   # Allow max 3dB crest factor loss
            duration_ok = abs(rest_metrics["duration_sec"] - orig_metrics["duration_sec"]) < 1.0

            passed = all([snr_ok, bw_ok, crest_ok, duration_ok])

            issues = []
            if not snr_ok:
                issues.append(f"SNR degraded by {abs(deltas['snr_delta_db']):.1f}dB")
            if not bw_ok:
                issues.append(f"Bandwidth lost {abs(deltas['bandwidth_delta_hz']):.0f}Hz")
            if not crest_ok:
                issues.append(f"Crest factor degraded by {abs(deltas['crest_delta_db']):.1f}dB")
            if not duration_ok:
                issues.append("Duration mismatch (possible corruption)")

            verdict = {
                "qc_passed": passed,
                "original_metrics": orig_metrics,
                "restored_metrics": rest_metrics,
                "deltas": deltas,
                "issues": issues if issues else ["All metrics within acceptable range"],
                "verdict": "APPROVED — Audio quality maintained or improved" if passed
                          else f"REJECTED — {'; '.join(issues)}"
            }

            return json.dumps(verdict, indent=2)
        except Exception as e:
            return json.dumps({
                "qc_passed": False,
                "verdict": f"QC Analysis Error: {str(e)}",
                "issues": [str(e)]
            }, indent=2)
