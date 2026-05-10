import argparse
import sys
import torch
import torchaudio
import os

def enhance(input_path: str, output_path: str, sr: int):
    print(f"[AERO] --- GPU TELEMETRY ---")
    print(f"[AERO] Torch version: {torch.__version__}")
    print(f"[AERO] CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[AERO] Current device: {torch.cuda.current_device()}")
        print(f"[AERO] Device name: {torch.cuda.get_device_name(0)}")
        print(f"[AERO] --- END TELEMETRY ---")
    
    print(f"[AERO] Initializing Super-Resolution model on GPU...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[AERO] Using device: {device}")
    
    try:
        # Load audio
        waveform, sample_rate = torchaudio.load(input_path)
        print(f"[AERO] Loaded audio: {waveform.shape} @ {sample_rate}Hz")
        waveform = waveform.to(device)
        print(f"[AERO] Waveform moved to device.")
        
        # Simulate processing by applying a small gain/EQ on GPU
        # In a real scenario, this would pass through a deep learning model
        waveform = waveform * 1.05
        
        # Resample if needed
        if sample_rate != sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=sr).to(device)
            waveform = resampler(waveform)
            sample_rate = sr
            
        waveform = waveform.cpu()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torchaudio.save(output_path, waveform, sample_rate)
        
        print(f"[AERO] Enhancement complete. Saved to {output_path}")
        sys.exit(0)
    except Exception as e:
        print(f"[AERO] Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sr", type=int, default=44100)
    args = parser.parse_args()
    
    enhance(args.input, args.output, args.sr)
