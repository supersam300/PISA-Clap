import torch
import librosa
from transformers import ClapModel, ClapProcessor
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from audio_recs.record_audio import record_audio

FS = 48000
AUDIO_DIR = os.path.join(os.path.dirname(__file__), "..", "audio_recs")
AUDIO_DIR = os.path.abspath(AUDIO_DIR)

model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = model.to(device)

candidate_labels = [ "kim jong un is the master of goon", "singing","clacking noise","tapping noise","speaking","low frequency noise"]
def clap_model(recording=None, audio=None):
    recordings = []
    if recording is not None:
        if isinstance(recording, list):
            recordings.extend(recording)
        else:
            recordings.append(recording)
    if audio is not None:
        recordings.append(audio)
        
    if not recordings:
        wav_files = [f for f in os.listdir(AUDIO_DIR) if f.endswith('.wav')]
        if wav_files:
            latest_file = max([os.path.join(AUDIO_DIR, f) for f in wav_files], key=os.path.getctime)
            recordings.append(os.path.basename(latest_file))
            
    results_all = []
    for current_rec in recordings:
        if os.path.isabs(current_rec):
            audio_path = current_rec
        else:
            audio_path = os.path.join(AUDIO_DIR, current_rec)
            
        try:
            audio_file, _ = librosa.load(audio_path, sr=FS, mono=True)
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            continue

        inputs = processor(
            text=candidate_labels,
            audio=[audio_file],
            return_tensors="pt",
            padding=True,
            sampling_rate=FS
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_audio = outputs.logits_per_audio
            probs = logits_per_audio.softmax(dim=1).squeeze().tolist()

            if not isinstance(probs, list):
                probs = [probs]
            res = list(zip(candidate_labels, probs))
            res.sort(key=lambda x: x[1], reverse=True)
            results_all.append(res)
            print(f"\nResults for {current_rec}:")
            for label, prob in res:
                print(f"  {label}: {prob:.4f}")

    if not results_all:
        return None

    if audio is not None and len(results_all) == 1:
        return "\n".join([f"{label}: {prob:.4f}" for label, prob in results_all[0]])
    return results_all[0] if len(results_all) == 1 else results_all

if __name__ == "__main__":
    recorded_file = record_audio()
    clap_model(recording=recorded_file)
    