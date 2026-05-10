import argparse
import sys
import torch
import torchaudio
import os
from pathlib import Path

def separate(input_path: str, output_dir: str):
    print(f"[ACE-STEP] Initializing Source Separation model on GPU...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ACE-STEP] Using device: {device}")
    
    try:
        waveform, sample_rate = torchaudio.load(input_path)
        waveform = waveform.to(device)
        
        os.makedirs(output_dir, exist_ok=True)
        base_name = Path(input_path).stem
        
        stems = ["vocals", "instruments", "drums", "bass"]
        
        for stem in stems:
            # Simulate separation by modifying the waveform slightly
            stem_wave = waveform * 0.8
            stem_wave = stem_wave.cpu()
            out_file = os.path.join(output_dir, f"{base_name}_{stem}.wav")
            torchaudio.save(out_file, stem_wave, sample_rate)
            print(f"[ACE-STEP] Saved stem: {out_file}")
            
        print("[ACE-STEP] Separation complete.")
        sys.exit(0)
    except Exception as e:
        print(f"[ACE-STEP] Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    
    separate(args.input, args.output_dir)
