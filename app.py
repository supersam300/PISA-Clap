import os
import glob
import numpy as np
import librosa
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import gradio as gr

# ── dataset helpers ──────────────────────────────────────────────────────────

DATASET_ROOT = os.path.join(os.path.dirname(__file__), "fan")

def get_machine_ids():
    return sorted([d for d in os.listdir(DATASET_ROOT)
                   if os.path.isdir(os.path.join(DATASET_ROOT, d))])

def get_files(machine_id: str, label: str):
    folder = os.path.join(DATASET_ROOT, machine_id, label)
    files = sorted(glob.glob(os.path.join(folder, "*.wav")))
    return [os.path.basename(f) for f in files]

def resolve_path(machine_id: str, label: str, filename: str) -> str:
    return os.path.join(DATASET_ROOT, machine_id, label, filename)

# ── audio loading ─────────────────────────────────────────────────────────────

def load_mono(path: str, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    """Load a wav file and return (mono_float32, sr)."""
    y, sr = sf.read(path, always_2d=True)          # shape (samples, channels)
    y = y.mean(axis=1).astype(np.float32)           # mix to mono
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    # peak-normalise so amplitude differences don't swamp spectral differences
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak
    return y, sr

# ── comparison core ───────────────────────────────────────────────────────────

N_MELS   = 128
HOP_LEN  = 512          # ~32 ms at 16 kHz  → fine time resolution
N_FFT    = 2048
SIGMA    = 1.5          # how many std-devs above mean = "fault"

def compute_mel(y: np.ndarray, sr: int) -> np.ndarray:
    S = librosa.feature.melspectrogram(y=y, sr=sr,
                                       n_fft=N_FFT, hop_length=HOP_LEN,
                                       n_mels=N_MELS, fmax=sr // 2)
    return librosa.power_to_db(S, ref=np.max)       # (n_mels, T)

def frame_diff_scores(mel1: np.ndarray, mel2: np.ndarray) -> np.ndarray:
    """Per-frame mean-absolute difference after aligning lengths."""
    T = min(mel1.shape[1], mel2.shape[1])
    d = np.mean(np.abs(mel1[:, :T] - mel2[:, :T]), axis=0)   # (T,)
    return d

def fault_spans(scores: np.ndarray, sr: int,
                threshold: float) -> list[tuple[float, float]]:
    """Return list of (t_start, t_end) where score exceeds threshold."""
    mask = scores > threshold
    spans = []
    in_span = False
    for i, m in enumerate(mask):
        t = librosa.frames_to_time(i, sr=sr, hop_length=HOP_LEN)
        if m and not in_span:
            t0, in_span = t, True
        elif not m and in_span:
            spans.append((t0, t))
            in_span = False
    if in_span:
        spans.append((t0, librosa.frames_to_time(len(mask), sr=sr,
                                                 hop_length=HOP_LEN)))
    # merge spans closer than 0.1 s
    merged = []
    for s in spans:
        if merged and s[0] - merged[-1][1] < 0.1:
            merged[-1] = (merged[-1][0], s[1])
        else:
            merged.append(list(s))
    return [(a, b) for a, b in merged]

# ── plotting ──────────────────────────────────────────────────────────────────

COLORS = {
    "ref_wave":   "#4C9BE8",
    "cmp_wave":   "#4C9BE8",
    "fault_span": "#FF4444",
    "diff_line":  "#FF8C00",
    "threshold":  "#CC0000",
}

def make_figure(y1: np.ndarray, y2: np.ndarray, sr: int,
                scores: np.ndarray, spans: list, threshold: float,
                label1: str, label2: str) -> plt.Figure:
    times1 = np.linspace(0, len(y1) / sr, len(y1))
    times2 = np.linspace(0, len(y2) / sr, len(y2))
    score_times = librosa.frames_to_time(np.arange(len(scores)),
                                         sr=sr, hop_length=HOP_LEN)

    fig, axes = plt.subplots(3, 1, figsize=(14, 9),
                             gridspec_kw={"height_ratios": [2, 2, 1.5]})
    fig.patch.set_facecolor("#1A1A2E")
    for ax in axes:
        ax.set_facecolor("#16213E")
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#444")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")

    def shade_faults(ax, ydata):
        ymin, ymax = np.min(ydata), np.max(ydata)
        pad = (ymax - ymin) * 0.05 or 0.05
        for t0, t1 in spans:
            ax.axvspan(t0, t1, color=COLORS["fault_span"], alpha=0.35, zorder=2)

    # --- waveform 1 ---
    axes[0].plot(times1, y1, color=COLORS["ref_wave"], lw=0.6, zorder=3)
    shade_faults(axes[0], y1)
    axes[0].set_title(f"Reference: {label1}")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_xlim(0, max(times1[-1], times2[-1]))

    # --- waveform 2 ---
    axes[1].plot(times2, y2, color=COLORS["cmp_wave"], lw=0.6, zorder=3)
    shade_faults(axes[1], y2)
    axes[1].set_title(f"Compared:  {label2}")
    axes[1].set_ylabel("Amplitude")
    axes[1].set_xlim(0, max(times1[-1], times2[-1]))

    # --- diff score ---
    axes[2].fill_between(score_times, scores, color=COLORS["diff_line"],
                         alpha=0.5, zorder=3)
    axes[2].plot(score_times, scores, color=COLORS["diff_line"], lw=1, zorder=4)
    axes[2].axhline(threshold, color=COLORS["threshold"],
                    lw=1.5, ls="--", label=f"threshold ({threshold:.1f} dB)")
    for t0, t1 in spans:
        axes[2].axvspan(t0, t1, color=COLORS["fault_span"], alpha=0.3, zorder=2)
    axes[2].set_title("Frame-wise Mel-Spectrogram Difference")
    axes[2].set_ylabel("Δ dB")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_xlim(0, score_times[-1])
    axes[2].legend(facecolor="#1A1A2E", labelcolor="white", fontsize=8)

    fault_patch = mpatches.Patch(color=COLORS["fault_span"], alpha=0.6,
                                 label="Detected fault region")
    fig.legend(handles=[fault_patch], loc="lower center",
               facecolor="#1A1A2E", labelcolor="white", fontsize=9,
               ncol=1, bbox_to_anchor=(0.5, 0.0))

    fig.tight_layout(rect=[0, 0.03, 1, 1])
    return fig

# ── text report ───────────────────────────────────────────────────────────────

def build_report(spans: list, scores: np.ndarray, sr: int) -> str:
    if not spans:
        return "✅ No significant differences detected between the two audio files."
    lines = [f"⚠️  **{len(spans)} fault region(s) detected:**\n"]
    for i, (t0, t1) in enumerate(spans, 1):
        i0 = librosa.time_to_frames(t0, sr=sr, hop_length=HOP_LEN)
        i1 = librosa.time_to_frames(t1, sr=sr, hop_length=HOP_LEN)
        i0 = max(0, min(i0, len(scores) - 1))
        i1 = max(0, min(i1, len(scores) - 1))
        peak = np.max(scores[i0:i1+1]) if i0 <= i1 else 0
        lines.append(f"  **Region {i}** — `{t0:.3f}s → {t1:.3f}s`  "
                     f"(duration: {t1-t0:.3f}s,  peak Δ: {peak:.1f} dB)")
    return "\n".join(lines)

# ── main compare function ─────────────────────────────────────────────────────

def compare(path1: str | None, path2: str | None,
            mid1: str, lbl1: str, fn1: str,
            mid2: str, lbl2: str, fn2: str,
            sigma: float):

    # resolve paths (uploaded file takes priority)
    if path1:
        p1, name1 = path1, os.path.basename(path1)
    else:
        p1 = resolve_path(mid1, lbl1, fn1)
        name1 = f"{mid1}/{lbl1}/{fn1}"

    if path2:
        p2, name2 = path2, os.path.basename(path2)
    else:
        p2 = resolve_path(mid2, lbl2, fn2)
        name2 = f"{mid2}/{lbl2}/{fn2}"

    try:
        y1, sr = load_mono(p1)
        y2, _  = load_mono(p2, target_sr=sr)
    except Exception as e:
        return None, f"❌ Failed to load audio: {e}"

    mel1 = compute_mel(y1, sr)
    mel2 = compute_mel(y2, sr)

    scores = frame_diff_scores(mel1, mel2)
    threshold = float(np.mean(scores) + sigma * np.std(scores))
    spans = fault_spans(scores, sr, threshold)

    fig = make_figure(y1, y2, sr, scores, spans, threshold, name1, name2)
    report = build_report(spans, scores, sr)
    return fig, report

# ── gradio UI ─────────────────────────────────────────────────────────────────

machine_ids = get_machine_ids()

def update_files(machine_id, label):
    files = get_files(machine_id, label)
    return gr.update(choices=files, value=files[0] if files else None)

with gr.Blocks(title="Audio Fault Comparator") as demo:
    gr.Markdown(
        "# 🔊 Audio Fault Comparator\n"
        "Compare two fan audio recordings and highlight temporal differences."
    )

    with gr.Row():
        # ── Audio 1 ──
        with gr.Column():
            gr.Markdown("### Audio 1 (Reference)")
            upload1  = gr.Audio(label="Upload (optional)", type="filepath",
                                sources=["upload"])
            mid1_dd  = gr.Dropdown(machine_ids, value=machine_ids[0] if machine_ids else None,
                                   label="Machine ID")
            lbl1_dd  = gr.Dropdown(["normal", "abnormal"], value="normal",
                                   label="Label")
            files1   = get_files(machine_ids[0], "normal") if machine_ids else []
            fn1_dd   = gr.Dropdown(files1, value=files1[0] if files1 else None,
                                   label="File")
            mid1_dd.change(update_files, [mid1_dd, lbl1_dd], fn1_dd)
            lbl1_dd.change(update_files, [mid1_dd, lbl1_dd], fn1_dd)

        # ── Audio 2 ──
        with gr.Column():
            gr.Markdown("### Audio 2 (Compare)")
            upload2  = gr.Audio(label="Upload (optional)", type="filepath",
                                sources=["upload"])
            mid2_dd  = gr.Dropdown(machine_ids, value=machine_ids[0] if machine_ids else None,
                                   label="Machine ID")
            lbl2_dd  = gr.Dropdown(["normal", "abnormal"], value="abnormal",
                                   label="Label")
            files2   = get_files(machine_ids[0], "abnormal") if machine_ids else []
            fn2_dd   = gr.Dropdown(files2, value=files2[0] if files2 else None,
                                   label="File")
            mid2_dd.change(update_files, [mid2_dd, lbl2_dd], fn2_dd)
            lbl2_dd.change(update_files, [mid2_dd, lbl2_dd], fn2_dd)

    with gr.Row():
        sigma_sl = gr.Slider(0.5, 4.0, value=1.5, step=0.1,
                             label="Sensitivity (σ above mean = fault threshold)",
                             info="Lower = more regions flagged")
        run_btn  = gr.Button("🔍  Compare", variant="primary", scale=0)

    out_plot   = gr.Plot(label="Waveform + Difference")
    out_report = gr.Markdown(label="Report")

    run_btn.click(
        fn=compare,
        inputs=[upload1, upload2,
                mid1_dd, lbl1_dd, fn1_dd,
                mid2_dd, lbl2_dd, fn2_dd,
                sigma_sl],
        outputs=[out_plot, out_report],
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
