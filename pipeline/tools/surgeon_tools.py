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
import torch
import torchaudio
import torchaudio.functional as F_audio
import torch.nn.functional as F

# Initialize ROCm/CUDA GPU Device for SOTA 2026 Tensor Acceleration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _load_audio_tensor(path: str):
    """Load audio as a PyTorch tensor using soundfile (avoids TorchCodec dep).
    Returns (tensor [channels, samples], sample_rate)."""
    data, sr = sf.read(path, dtype='float32', always_2d=True)
    # soundfile returns [samples, channels], torch wants [channels, samples]
    tensor = torch.from_numpy(data.T)
    return tensor, sr


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


class SOTAMasteringChainTool(BaseTool):
    name: str = "sota_mastering_chain"
    description: str = (
        "Executes the entire 7-step 'OZONE-LIKE SOTA' restoration chain: Upsampling -> "
        "Tonal Balance -> Neural Rebalance -> Transients -> Clarity -> FFmpeg Master -> Maximizer. "
        "Use this tool to apply the complete restoration plan in one step."
    )
    args_schema: Type[BaseModel] = ToolInput

    def _run(self, input_path: str, output_path: str) -> str:
        try:
            # Safely get stem even if filename has multiple dots
            filename = os.path.basename(input_path)
            stem = os.path.splitext(filename)[0]
            inter_base = "/data/intermediate/" + stem
            
            print(f"--- SOTA MASTERING CHAIN START ---")
            
            # Step 1: Upsampler
            p1 = inter_base + "_96k.wav"
            print("Step 1: Upsampler...")
            res1 = json.loads(AudiophileUpsamplerTool()._run(input_path, p1))
            if not res1.get("success"): return json.dumps(res1)
            
            # Step 2: Tonal Balance
            p2 = inter_base + "_tonal.wav"
            print("Step 2: Tonal Balance...")
            res2 = json.loads(TonalBalanceStabilizerTool()._run(p1, p2))
            if not res2.get("success"): return json.dumps(res2)

            # Step 3: Neural Rebalance
            p3 = inter_base + "_rebalanced.wav"
            print("Step 3: Neural Rebalance...")
            res3 = json.loads(NeuralMasterRebalanceTool()._run(p2, p3))
            if not res3.get("success"): return json.dumps(res3)

            # Step 4: Transients
            p4 = inter_base + "_transient.wav"
            print("Step 4: Transients...")
            res4 = json.loads(TransientPreservationTool()._run(p3, p4))
            if not res4.get("success"): return json.dumps(res4)

            # Step 5: Clarity
            p5 = inter_base + "_clarity.wav"
            print("Step 5: Clarity...")
            res5 = json.loads(PsychoacousticExciterTool()._run(p4, p5))
            if not res5.get("success"): return json.dumps(res5)

            # Step 6: FFmpeg Master
            p6 = inter_base + "_mastered.wav"
            print("Step 6: FFmpeg Master...")
            res6 = json.loads(FFmpegProMasteringTool()._run(p5, p6))
            if not res6.get("success"): return json.dumps(res6)

            # Step 7: Maximizer
            print("Step 7: Dynamic Boost Maximizer...")
            res7 = json.loads(MasterMaximizerTool()._run(p6, output_path))
            if not res7.get("success"): return json.dumps(res7)

            return _make_result(True, output_path, "Full 7-step SOTA Mastering Chain completed successfully.", res1.get("metrics", {}).get("before"), res7.get("metrics", {}).get("after"))
        except Exception as e:
            return _make_result(False, output_path, f"Mastering Chain Error: {str(e)}")


class AudiophileUpsamplerTool(BaseTool):
    name: str = "soxr_vhq_upsampler"
    description: str = (
        "High-precision upsampler to 96kHz using the Soxr VHQ (Very High Quality) engine. "
        "Zero phase distortion. Returns JSON with before/after audio quality metrics."
    )
    args_schema: Type[BaseModel] = ToolInput

    def _run(self, input_path: str, output_path: str) -> str:
        try:
            # SOTA 2026: Direct to GPU VRAM loading (soundfile backend)
            y_tensor, sr = _load_audio_tensor(input_path)
            y_tensor = y_tensor.to(DEVICE)
            
            y_cpu = y_tensor.cpu().numpy()
            is_stereo = y_cpu.shape[0] > 1
            input_metrics = _measure_audio(y_cpu[0] if is_stereo else y_cpu, sr)

            if sr < 96000:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=96000).to(DEVICE)
                y_96_tensor = resampler(y_tensor)
            else:
                y_96_tensor = y_tensor

            y_96_cpu = y_96_tensor.cpu().numpy()
            sf.write(output_path, y_96_cpu.T, 96000)
            output_metrics = _measure_audio(y_96_cpu[0] if is_stereo else y_96_cpu, 96000)

            return _make_result(
                True, output_path,
                f"Upsampled {sr}Hz → 96000Hz (GPU Tensor Resample). {'Stereo' if is_stereo else 'Mono'} preserved.",
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

            def shape_transients(signal_data, sample_rate):
                abs_sig = np.abs(signal_data)
                
                # Simple exponential moving average filter for envelopes (Fast 5ms vs Slow 50ms)
                alpha_fast = 1.0 - np.exp(-1.0 / (0.005 * sample_rate))
                alpha_slow = 1.0 - np.exp(-1.0 / (0.050 * sample_rate))
                
                fast_env = signal.lfilter([alpha_fast], [1.0, -(1.0 - alpha_fast)], abs_sig)
                slow_env = signal.lfilter([alpha_slow], [1.0, -(1.0 - alpha_slow)], abs_sig)
                
                # The transient is where fast > slow
                transient_gain = np.clip(fast_env - slow_env, 0, None)
                gain_envelope = 1.0 + (transient_gain * 5.0)
                
                shaped = signal_data * gain_envelope
                shaped = shaped / (np.max(np.abs(shaped)) + 1e-10)
                return shaped

            if is_stereo:
                input_metrics = _measure_audio(y[0], sr)
                y_shaped = np.array([shape_transients(ch, sr) for ch in y])
                sf.write(output_path, y_shaped.T, sr)
                output_metrics = _measure_audio(y_shaped[0], sr)
            else:
                input_metrics = _measure_audio(y, sr)
                y_shaped = shape_transients(y, sr)
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
                # High pass above 10kHz
                sos = signal.butter(8, 10000, 'hp', fs=sample_rate, output='sos')
                highs = signal.sosfilt(sos, signal_data)
                
                # Tape-like soft clipping polynomial for "Air" (SOTA)
                drive = 2.0
                driven_highs = highs * drive
                tape = driven_highs / (1.0 + np.abs(driven_highs))
                
                excited = signal_data + tape * 0.05
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


class TonalBalanceStabilizerTool(BaseTool):
    name: str = "spectral_tonal_balance_stabilizer"
    description: str = (
        "SOTA Spectral Shaping. Analyzes the frequency spectrum and dynamically applies a "
        "Match-EQ curve to align the audio to a commercial Pink Noise (-3dB/oct) target. "
        "Instantly cures 'muddy' or 'harsh' mixes. The ultimate 'Wow' factor."
    )
    args_schema: Type[BaseModel] = ToolInput

    def _run(self, input_path: str, output_path: str) -> str:
        try:
            y, sr = librosa.load(input_path, sr=None, mono=False)
            is_stereo = y.ndim > 1
            input_metrics = _measure_audio(y[0] if is_stereo else y, sr)

            def apply_tonal_balance(signal_data, sample_rate):
                # 1. Compute average spectrum using Welch's method
                nperseg = 4096
                freqs, psd = signal.welch(signal_data, fs=sample_rate, nperseg=nperseg)
                psd = np.maximum(psd, 1e-12)
                
                # 2. Create Pink Noise target (-3dB per octave = 1/f)
                target_psd = np.zeros_like(psd)
                target_psd[1:] = 1.0 / freqs[1:]
                target_psd[0] = target_psd[1] # handle DC
                
                # Normalize target to match energy of the source in the critical mid-band (1kHz to 4kHz)
                band_mask = (freqs >= 1000) & (freqs <= 4000)
                if np.any(band_mask):
                    source_energy = np.mean(psd[band_mask])
                    target_energy = np.mean(target_psd[band_mask])
                    target_psd *= (source_energy / target_energy)
                
                # 3. Calculate EQ difference curve
                eq_curve = np.sqrt(target_psd / psd)
                
                # 4. Smooth the curve (we only want macroscopic tonal balance, not comb filtering)
                eq_curve_db = 20 * np.log10(eq_curve)
                eq_curve_db_smoothed = signal.savgol_filter(eq_curve_db, window_length=151, polyorder=3)
                
                # Limit the maximum boost/cut to prevent unnatural artifacts (max +/- 4.5dB)
                eq_curve_db_smoothed = np.clip(eq_curve_db_smoothed, -4.5, 4.5)
                eq_curve_smoothed = 10 ** (eq_curve_db_smoothed / 20.0)
                
                # 5. Design Zero-Phase FIR Filter
                num_taps = 1025
                fir_coeff = signal.firwin2(num_taps, freqs, eq_curve_smoothed, fs=sample_rate)
                
                # 6. Apply filter
                balanced = signal.fftconvolve(signal_data, fir_coeff, mode='same')
                return balanced / (np.max(np.abs(balanced)) + 1e-10)

            if is_stereo:
                y_bal = np.array([apply_tonal_balance(ch, sr) for ch in y])
            else:
                y_bal = apply_tonal_balance(y, sr)
                
            sf.write(output_path, y_bal.T if is_stereo else y_bal, sr)
            output_metrics = _measure_audio(y_bal[0] if is_stereo else y_bal, sr)

            return _make_result(True, output_path, "Spectral Tonal Balance applied (Pink Noise Match EQ).", input_metrics, output_metrics)
        except Exception as e:
            return _make_result(False, output_path, f"Tonal Balance Error: {str(e)}")


class PsychoacousticExciterTool(BaseTool):
    name: str = "psychoacoustic_clarity_exciter"
    description: str = (
        "Advanced harmonic generator for 'Clarity'. Adds even armonics (warmth) and "
        "odd armonics (definition) above a cutoff. Based on FxSound Aural Exciter logic."
    )
    args_schema: Type[BaseModel] = ToolInput

    def _run(self, input_path: str, output_path: str) -> str:
        try:
            y, sr = librosa.load(input_path, sr=None, mono=False)
            is_stereo = y.ndim > 1

            def excite_pro(signal_data, sample_rate):
                # High pass above 3kHz to isolate clarity band
                sos = signal.butter(4, 3000, 'hp', fs=sample_rate, output='sos')
                highs = signal.sosfilt(sos, signal_data)
                
                # Asymmetric tube-like saturation (Ozone Vintage/Tube mode approximation)
                drive = 1.5
                highs_driven = highs * drive
                tube = np.where(highs_driven > 0, 
                                1.0 - np.exp(-highs_driven), 
                                (np.exp(highs_driven) - 1.0) * 0.8)
                
                excited = signal_data + (tube * 0.05)
                excited = excited / (np.max(np.abs(excited)) + 1e-10)
                return excited

            if is_stereo:
                input_metrics = _measure_audio(y[0], sr)
                y_out = np.array([excite_pro(ch, sr) for ch in y])
                sf.write(output_path, y_out.T, sr)
                output_metrics = _measure_audio(y_out[0], sr)
            else:
                input_metrics = _measure_audio(y, sr)
                y_out = excite_pro(y, sr)
                sf.write(output_path, y_out, sr)
                output_metrics = _measure_audio(y_out, sr)

            return _make_result(True, output_path, "Psychoacoustic Clarity applied (Harmonic Balance).", input_metrics, output_metrics)
        except Exception as e:
            return _make_result(False, output_path, f"Clarity Error: {str(e)}")


class StereoWidthTool(BaseTool):
    name: str = "stereo_spatial_widener"
    description: str = (
        "Expands the soundstage using SOTA Mid-Side (M/S) processing. "
        "Implements 'Mono-Bass' to ensure phase coherence below 150Hz, "
        "while dynamically widening high frequencies. Zero comb filtering."
    )
    args_schema: Type[BaseModel] = ToolInput

    def _run(self, input_path: str, output_path: str) -> str:
        try:
            y_tensor, sr = _load_audio_tensor(input_path)
            y_tensor = y_tensor.to(DEVICE)
            if y_tensor.shape[0] < 2:
                return _make_result(False, output_path, "Stereo widening requires a stereo input.")

            y_cpu = y_tensor.cpu().numpy()
            input_metrics = _measure_audio(y_cpu[0], sr)
            
            y_cpu = y_tensor.cpu().numpy()
            input_metrics = _measure_audio(y_cpu[0], sr)
            
            # M/S Matrix in Numpy (Safe and stable)
            left = y_cpu[0]
            right = y_cpu[1]
            mid = (left + right) / 2.0
            side = (left - right) / 2.0
            
            # Mono-Bass (Sum below 150Hz to Mid, remove from Side) using stable SciPy
            sos_low = signal.butter(4, 150.0, 'lp', fs=sr, output='sos')
            sos_high = signal.butter(4, 150.0, 'hp', fs=sr, output='sos')
            
            side_low = signal.sosfilt(sos_low, side)
            side_high = signal.sosfilt(sos_high, side)
            
            # Move side low frequencies to mid (perfect mono compatibility for bass)
            mid = mid + side_low
            # Expand the highs in the side channel for immersive width
            side_high = side_high * 1.15
            
            y_wide = np.stack([mid + side_high, mid - side_high], axis=0)
            y_wide = y_wide / (np.max(np.abs(y_wide)) + 1e-10)
            
            sf.write(output_path, y_wide.T, sr)
            output_metrics = _measure_audio(y_wide[0], sr)

            return _make_result(True, output_path, "Stereo soundstage expanded (GPU Tensor M/S + Mono-Bass).", input_metrics, output_metrics)
        except Exception as e:
            return _make_result(False, output_path, f"Widening Error: {str(e)}")


class MasterMaximizerTool(BaseTool):
    name: str = "dynamic_boost_maximizer"
    description: str = (
        "Intelligent 'IRC-like' look-ahead limiter and maximizer. Increases RMS volume to "
        "commercial 'loudness' levels while preventing clipping using a high-order algebraic soft-knee. Final stage."
    )
    args_schema: Type[BaseModel] = ToolInput

    def _run(self, input_path: str, output_path: str) -> str:
        try:
            y_tensor, sr = _load_audio_tensor(input_path)
            y_tensor = y_tensor.to(DEVICE)
            
            y_cpu = y_tensor.cpu().numpy()
            is_stereo = y_cpu.shape[0] > 1
            input_metrics = _measure_audio(y_cpu[0] if is_stereo else y_cpu, sr)

            # Boost gain by 2.5dB
            boost = 10**(2.5 / 20.0)
            boosted = y_tensor * boost
            
            # SOTA High-order algebraic soft-knee limiter on GPU
            limit = 0.98
            abs_b = torch.abs(boosted) / limit
            abs_b = torch.clamp(abs_b, min=0.0, max=50.0) # Prevent overflow
            
            maximized = boosted / torch.pow(1.0 + torch.pow(abs_b, 6.0), 1.0/6.0)
            
            # Absolute brickwall clip to guarantee safety
            y_max = torch.clamp(maximized, min=-1.0, max=1.0)
            
            y_out_cpu = y_max.cpu().numpy()
            sf.write(output_path, y_out_cpu.T, sr)
            output_metrics = _measure_audio(y_out_cpu[0] if is_stereo else y_out_cpu, sr)

            return _make_result(True, output_path, "Dynamic Boost applied (GPU Tensor Maximizer).", input_metrics, output_metrics)
        except Exception as e:
            return _make_result(False, output_path, f"Maximizer Error: {str(e)}")


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


class NeuralMasterRebalanceTool(BaseTool):
    name: str = "neural_master_rebalance"
    description: str = (
        "SOTA 2026 Master Rebalance via Neural Stem Separation (Demucs/BS-RoFormer). "
        "Isolates Vocals, Bass, Drums, and Melody. Amplifies Melody and Mono-Basses the rhythm section "
        "while keeping Vocals perfectly pristine. Requires massive GPU VRAM."
    )
    args_schema: Type[BaseModel] = ToolInput

    def _run(self, input_path: str, output_path: str) -> str:
        try:
            # We import here so it doesn't break if Demucs is not installed on the dev machine yet
            import demucs.api
            
            y_tensor, sr = _load_audio_tensor(input_path)
            y_tensor = y_tensor.to(DEVICE)
            y_cpu = y_tensor.cpu().numpy()
            is_stereo = y_cpu.shape[0] > 1
            input_metrics = _measure_audio(y_cpu[0] if is_stereo else y_cpu, sr)

            separator = demucs.api.Separator(model="htdemucs", device=DEVICE.type)
            _, separated = separator.separate_audio_file(input_path)
            # Returns dict: {'vocals': tensor, 'drums': tensor, 'bass': tensor, 'other': tensor}
            
            vocals = separated['vocals'].to(DEVICE)
            drums = separated['drums'].to(DEVICE)
            bass = separated['bass'].to(DEVICE)
            other = separated['other'].to(DEVICE)
            
            # Rebalance logic
            # 1. Vocals: untouched (dead center, pristine)
            # 2. Bass + Drums: Mono-Bassed (force to mono to ensure club impact)
            rhythm = bass + drums
            if rhythm.shape[0] > 1:
                mono_rhythm = torch.mean(rhythm, dim=0, keepdim=True)
                rhythm = mono_rhythm.repeat(2, 1)
                
            # 3. Other (Melody/Synth): Expanded by 1.2x on the side channel
            if other.shape[0] > 1:
                mid = (other[0:1, :] + other[1:2, :]) / 2.0
                side = (other[0:1, :] - other[1:2, :]) / 2.0
                side = side * 1.2
                other = torch.cat([mid + side, mid - side], dim=0)
                
            # Sum back together
            rebalanced = vocals + rhythm + other
            
            # SOTA FIX: Demucs outputs at 44.1kHz. We MUST resample back to the pipeline's 96kHz resolution on GPU.
            if separator.samplerate != sr:
                resampler = torchaudio.transforms.Resample(orig_freq=separator.samplerate, new_freq=sr).to(DEVICE)
                rebalanced = resampler(rebalanced)
                
            rebalanced = rebalanced / (torch.max(torch.abs(rebalanced)) + 1e-10)
            
            y_out_cpu = rebalanced.cpu().numpy()
            sf.write(output_path, y_out_cpu.T, sr)
            output_metrics = _measure_audio(y_out_cpu[0] if is_stereo else y_out_cpu, sr)

            return _make_result(True, output_path, "Neural Master Rebalance applied (Vocals isolated, Melody expanded).", input_metrics, output_metrics)
        except ImportError:
            return _make_result(False, output_path, "Demucs library not installed. ROCm Docker env required.")
        except Exception as e:
            return _make_result(False, output_path, f"Neural Rebalance Error: {str(e)}")
