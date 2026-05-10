import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Dict, Any

class SpectralAnalyzer:
    """Detects audio quality issues using DSP and Spectral analysis."""
    
    def __init__(self, sr=44100):
        self.target_sr = sr

    def analyze(self, audio_path: str) -> Dict[str, Any]:
        """Runs a full spectral diagnostic on a file."""
        y, sr = librosa.load(audio_path, sr=None)
        
        # 1. Detect Frequency Cutoff (The 'Suno Ceiling')
        # We look for the frequency where energy drops significantly
        stft = np.abs(librosa.stft(y))
        freqs = librosa.fft_frequencies(sr=sr)
        avg_spectrum = np.mean(stft, axis=1)
        
        # Normalize spectrum
        avg_spectrum_db = librosa.amplitude_to_db(avg_spectrum)
        
        # Find the point where energy drops below -60dB relative to max
        max_db = np.max(avg_spectrum_db)
        threshold = max_db - 50
        cutoff_idx = np.where(avg_spectrum_db < threshold)[0]
        
        cutoff_freq = freqs[cutoff_idx[0]] if len(cutoff_idx) > 0 else sr/2
        
        # 2. Estimate Noise Floor (using silence detection)
        # Assuming quiet parts are noise
        intervals = librosa.effects.split(y, top_db=40)
        noise_segments = []
        last_end = 0
        for start, end in intervals:
            if start > last_end:
                noise_segments.append(y[last_end:start])
            last_end = end
            
        noise_floor = -100
        if noise_segments:
            noise_concatenated = np.concatenate(noise_segments)
            if len(noise_concatenated) > 0:
                noise_floor = librosa.amplitude_to_db([np.sqrt(np.mean(noise_concatenated**2))])[0]

        # 3. Dynamic Range (Peak to RMS)
        rms = librosa.feature.rms(y=y)[0]
        peak = np.max(np.abs(y))
        avg_rms = np.mean(rms)
        crest_factor = 20 * np.log10(peak / avg_rms) if avg_rms > 0 else 0

        return {
            "filename": Path(audio_path).name,
            "sample_rate": sr,
            "duration_sec": librosa.get_duration(y=y, sr=sr),
            "cutoff_freq_hz": int(cutoff_freq),
            "noise_floor_db": float(noise_floor),
            "crest_factor_db": float(crest_factor),
            "needs_super_res": bool(cutoff_freq < 16000),
            "needs_denoising": bool(noise_floor > -50)
        }

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        analyzer = SpectralAnalyzer()
        report = analyzer.analyze(sys.argv[1])
        print(report)
