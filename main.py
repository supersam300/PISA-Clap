import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

AUDIO_DIR = os.path.abspath(os.path.join(_PROJECT_ROOT, "audio_recs"))
ANALYZE_TIMEOUT_SEC = float(os.getenv("ANALYZE_TIMEOUT_SEC", "30"))
GRADIO_SERVER_NAME = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
_gradio_port_env = os.getenv("GRADIO_SERVER_PORT")
if _gradio_port_env is None or _gradio_port_env.strip() == "":

    GRADIO_SERVER_PORT = None
else:
    try:
        GRADIO_SERVER_PORT = int(_gradio_port_env)
    except ValueError:
        GRADIO_SERVER_PORT = None

_analyze_maintenance = None


def _get_analyzer():
    global _analyze_maintenance
    if _analyze_maintenance is not None:
        return _analyze_maintenance
    from model.maintenance import analyze_maintenance
    _analyze_maintenance = analyze_maintenance
    return _analyze_maintenance

def gradio_maintenance(audio):
    try:
        analyze = _get_analyzer()
    except Exception as e:
        msg = f"Failed to initialize maintenance analyzer: {e}"
        return msg, "UNKNOWN", "N/A", msg, [], {"error": msg}

    if not audio:
        msg = "Please upload or record an audio clip."
        return msg, "UNKNOWN", "N/A", msg, [], {"error": msg}

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(analyze, audio)
            report = future.result(timeout=ANALYZE_TIMEOUT_SEC)
    except FutureTimeoutError:
        msg = (
            f"Analysis timed out after {ANALYZE_TIMEOUT_SEC:.0f}s. "
            "Try a shorter clip or disable Gemini diagnosis."
        )
        return msg, "UNKNOWN", "N/A", msg, [], {"error": msg}
    except Exception as e:
        msg = f"Failed to analyze audio: {e}"
        return msg, "UNKNOWN", "N/A", msg, [], {"error": msg}
    issue = "YES" if report["issue_detected"] else "NO"
    status_md = (
        f"### Maintenance Decision\n"
        f"- **Issue Detected:** `{issue}`\n"
        f"- **Priority:** `{report['maintenance_priority']}`"
    )
    top_signal = f"{report['top_label']} ({report['top_score']:.4f})"
    ranking_table = [[label, round(score, 4)] for label, score in report["ranked_results"]]

    return (
        status_md,
        report["maintenance_priority"],
        top_signal,
        report["description"],
        ranking_table,
        report,
    )


def build_gradio_app(gr):
    with gr.Blocks(title="Maintenance Analyzer") as demo:
        gr.Markdown(
            "# Maintenance Analyzer\n"
            "Upload or record machine audio to get issue detection, maintenance priority, and diagnosis."
        )

        with gr.Row():
            audio_input = gr.Audio(
                type="filepath",
                label="Input Audio",
                sources=["upload", "microphone"],
                editable=False,
                waveform_options=gr.WaveformOptions(sample_rate=16000),
            )
            with gr.Column():
                analyze_btn = gr.Button("Analyze Audio", variant="primary")
                clear_btn = gr.Button("Clear")

        with gr.Row():
            status_md = gr.Markdown(label="Maintenance Decision")
            with gr.Column():
                priority_box = gr.Textbox(label="Priority", interactive=False)
                top_signal_box = gr.Textbox(label="Top Signal", interactive=False)

        diagnosis_box = gr.Textbox(
            label="Diagnosis",
            lines=6,
            interactive=False,
        )
        ranking_df = gr.Dataframe(
            headers=["CLAP Label", "Score"],
            datatype=["str", "number"],
            label="CLAP Ranking",
            interactive=False,
        )
        raw_json = gr.JSON(label="Raw Analysis Output")

        analyze_btn.click(
            fn=gradio_maintenance,
            inputs=[audio_input],
            outputs=[status_md, priority_box, top_signal_box, diagnosis_box, ranking_df, raw_json],
        )

        clear_btn.click(
            fn=lambda: ("", "", "", "", [], {}),
            inputs=[],
            outputs=[status_md, priority_box, top_signal_box, diagnosis_box, ranking_df, raw_json],
        )
        
        custom_lables = gr.Dropdown(
            label="Custom Labels",
            choices=["label1", "label2", "label3"],
            interactive=True,
        )

    return demo

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--loop", action="store_true", help="Run in continuous recording loop instead of Gradio")
    args = parser.parse_args()

    if args.loop:
        try:
            from audio_recs.record_audio import record_audio
            from model.maintenance import analyze_maintenance, format_maintenance_report
        except ModuleNotFoundError as e:
            print(f"Missing dependency: {e.name}. Install requirements from req.txt and retry.")
            raise SystemExit(1)
        except Exception as e:
            print(f"Startup error: {e}")
            raise SystemExit(1)

        while True:
            try:
                recorded_file = record_audio()
                report = analyze_maintenance(recorded_file)
                print(format_maintenance_report(report))

                time.sleep(10)   

                for files in os.listdir(AUDIO_DIR):
                    if files.endswith(".wav"):
                        os.remove(os.path.join(AUDIO_DIR, files))         
            except KeyboardInterrupt:
                print("\nStopped by user.")
                break
            except Exception as e:
                print(f"Error: {e}")
                break
    else:
        try:
            import gradio as gr
        except ModuleNotFoundError:
            print("Missing dependency: gradio. Install it with `pip install gradio`.")
            raise SystemExit(1)

        print("Starting Gradio Interface. Use --loop to run continuous recording.")
        demo = build_gradio_app(gr)
        launch_kwargs = {
            "max_file_size": "50mb",
            "server_name": GRADIO_SERVER_NAME,
        }
        if GRADIO_SERVER_PORT is not None:
            launch_kwargs["server_port"] = GRADIO_SERVER_PORT
        demo.launch(**launch_kwargs)
