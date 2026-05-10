import librosa
import glob
import os

files = glob.glob("/data/intermediate/*.wav") + ["/data/output/C_restored.wav", "/data/input/C.wav"]
print(f"{'Filename':<30} | {'Duration (sec)':<15} | {'Sample Rate'}")
print("-" * 70)
for f in sorted(files):
    if os.path.exists(f):
        y, sr = librosa.load(f, sr=None)
        print(f"{os.path.basename(f):<30} | {len(y)/sr:<15.3f} | {sr} Hz")
