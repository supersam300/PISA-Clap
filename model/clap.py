import os
import sys
from typing import List, Sequence, Tuple

import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ClapModel, ClapProcessor

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
_PUMP_WEIGHTS = os.path.join(_MODEL_DIR, "jedi", "pump_classifier.pth")
ABNORMAL_TRIGGER_THRESHOLD = 0.25

FS = 48000
MODEL_NAME = "laion/clap-htsat-unfused"
AUDIO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "audio_recs"))

_processor = None
_model = None

_bin_model = None

## build a pass to classify object
##this is done
## build a function to switch FFNN?????


class PfaultClassifier(nn.Module):
    def __init__(self, ebbeding_dim=512):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(ebbeding_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        return self.network(x)


def _load_bin_model():
    global _bin_model
    if _bin_model is not None:
        return _bin_model
    if not os.path.isfile(_PUMP_WEIGHTS):
        return None
    try:
        device = get_device()
        m = PfaultClassifier().to(device)
        m.load_state_dict(torch.load(_PUMP_WEIGHTS, map_location=device))
        m.eval()
        _bin_model = m
        return _bin_model
    except Exception:
        return None


NORMAL_LABEL = "normal machine operating hum"
ABNORMAL_CANDIDATE_LABELS = [
    "continuous low-frequency rumble",
    "intermittent metallic clacking",
    "repetitive tapping or knocking",
    "high-pitched squeal from friction",
    "air leak or hissing noise",
    "grinding noise from worn bearings",
]
candidate_labels = [NORMAL_LABEL, *ABNORMAL_CANDIDATE_LABELS]


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _get_processor_and_model() -> Tuple[ClapProcessor, ClapModel]:
    global _processor, _model
    if _processor is None:
        _processor = ClapProcessor.from_pretrained(MODEL_NAME)
    if _model is None:
        _model = ClapModel.from_pretrained(MODEL_NAME).to(get_device())
        _model.eval()
    return _processor, _model


def load_audio_48k_mono(audio_path: str):
    import subprocess
    import tempfile
    import shutil

    try:
        return librosa.load(audio_path, sr=FS, mono=True)
    except Exception:
        pass

    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        raise RuntimeError(f"Cannot decode {audio_path}: librosa failed and ffmpeg not found")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        subprocess.run(
            [ffmpeg_bin, "-y", "-i", audio_path, "-ar", str(FS), "-ac", "1", tmp_path],
            capture_output=True,
            timeout=15,
            check=True,
        )
        return librosa.load(tmp_path, sr=FS, mono=True)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def get_audio_embedding(audio_array):
    processor, model = _get_processor_and_model()
    device = get_device()
    inputs = processor(audio=[audio_array], sampling_rate=FS, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        emb_out = model.get_audio_features(**inputs)
    if isinstance(emb_out, torch.Tensor):
        emb = emb_out
    elif hasattr(emb_out, "pooler_output"):
        emb = emb_out.pooler_output
    elif hasattr(emb_out, "last_hidden_state"):
        emb = emb_out.last_hidden_state.mean(dim=1)
    else:
        raise TypeError(f"Unexpected audio embedding output type: {type(emb_out)}")
    return F.normalize(emb, p=2, dim=-1)


def get_text_embeddings(texts: Sequence[str]):
    processor, model = _get_processor_and_model()
    device = get_device()
    inputs = processor(text=list(texts), return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        emb_out = model.get_text_features(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )
    if isinstance(emb_out, torch.Tensor):
        emb = emb_out
    elif hasattr(emb_out, "pooler_output"):
        emb = emb_out.pooler_output
    elif hasattr(emb_out, "last_hidden_state"):
        emb = emb_out.last_hidden_state.mean(dim=1)
    else:
        raise TypeError(f"Unexpected text embedding output type: {type(emb_out)}")
    return F.normalize(emb, p=2, dim=-1)


def cosine_similarity_scores(audio_embedding: torch.Tensor, text_embeddings: torch.Tensor):
    return torch.matmul(audio_embedding, text_embeddings.T).squeeze(0)


def rank_text_for_audio(audio_array, texts: Sequence[str]) -> List[Tuple[str, float]]:
    if not texts:
        return []
    audio_embedding = get_audio_embedding(audio_array)
    text_embeddings = get_text_embeddings(texts)
    scores = cosine_similarity_scores(audio_embedding, text_embeddings).tolist()
    ranked = list(zip(texts, scores))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


def _resolve_recordings(recording=None, audio=None) -> List[str]:
    recordings: List[str] = []
    if recording is not None:
        if isinstance(recording, list):
            recordings.extend(recording)
        else:
            recordings.append(recording)
    if audio is not None:
        recordings.append(audio)
    if recordings:
        return recordings

    wav_files = [f for f in os.listdir(AUDIO_DIR) if f.endswith(".wav")]
    if wav_files:
        latest_file = max(
            [os.path.join(AUDIO_DIR, f) for f in wav_files], key=os.path.getctime
        )
        recordings.append(os.path.basename(latest_file))
    return recordings


def resolve_audio_path(recording_name: str) -> str:
    return recording_name if os.path.isabs(recording_name) else os.path.join(AUDIO_DIR, recording_name)


def classify_audio_path(audio_path: str, labels: Sequence[str] = None) -> List[Tuple[str, float]]:
    labels = list(labels) if labels is not None else candidate_labels
    audio_file, _ = load_audio_48k_mono(audio_path)
    return rank_text_for_audio(audio_file, labels)


def is_faulty(audio_path: str) -> int:
    """Return 1 if FFNN predicts abnormal, 0 if normal."""
    bin_model = _load_bin_model()
    if bin_model is None:
        return 1
    audio_file, _ = load_audio_48k_mono(audio_path)
    audio_vector = get_audio_embedding(audio_file)
    with torch.no_grad():
        logits = bin_model(audio_vector)
        probs = torch.softmax(logits, dim=1)
        abnormal_prob = float(probs[:, 1].item())
        return 1 if abnormal_prob >= ABNORMAL_TRIGGER_THRESHOLD else 0


def classify_audio_path_two_pass(
    audio_path: str, abnormal_labels: Sequence[str] = None
) -> dict:
    """Two-pass inference: FFNN gate first, CLAP ranking only for abnormal sounds."""
    prediction = is_faulty(audio_path)
    if prediction == 0:
        return {
            "is_abnormal": False,
            "ranked_results": [(NORMAL_LABEL, 1.0)],
            "top_label": NORMAL_LABEL,
            "top_score": 1.0,
            "stage": "ffnn_only",
        }

    labels = list(abnormal_labels) if abnormal_labels is not None else ABNORMAL_CANDIDATE_LABELS
    ranked = classify_audio_path(audio_path, labels=labels)
    if not ranked:
        raise ValueError("No CLAP result produced for abnormal audio.")
    return {
        "is_abnormal": True,
        "ranked_results": ranked,
        "top_label": ranked[0][0],
        "top_score": ranked[0][1],
        "stage": "ffnn_then_clap",
    }


def clap_model(recording=None, audio=None):
    recordings = _resolve_recordings(recording=recording, audio=audio)
    results_all = []

    for current_rec in recordings:
        audio_path = resolve_audio_path(current_rec)
        try:
            pass_result = classify_audio_path_two_pass(audio_path)
            res = pass_result["ranked_results"]
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            continue

        results_all.append(res)
        print(f"\nResults for {current_rec}:")
        for label, score in res:
            print(f"  {label}: {score:.4f}")

    if not results_all:
        return None
    if audio is not None and len(results_all) == 1:
        return "\n".join([f"{label}: {score:.4f}" for label, score in results_all[0]])
    return results_all[0] if len(results_all) == 1 else results_all


if __name__ == "__main__":
    from audio_recs.record_audio import record_audio

    recorded_file = record_audio()
    clap_model(recording=recorded_file)
