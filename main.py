import os
import time

AUDIO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "audio_recs"))


def gradio_maintenance(audio):
    try:
        from model.maintenance import analyze_maintenance
    except ModuleNotFoundError as e:
        msg = f"Missing dependency: {e.name}. Install requirements and try again."
        return msg, "UNKNOWN", "N/A", msg, [], {"error": msg}
    except Exception as e:
        msg = f"Failed to initialize maintenance analyzer: {e}"
        return msg, "UNKNOWN", "N/A", msg, [], {"error": msg}

    if not audio:
        msg = "Please upload or record an audio clip."
        return msg, "UNKNOWN", "N/A", msg, [], {"error": msg}

    try:
        report = analyze_maintenance(audio)
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
    with gr.Blocks(title="CLAP Maintenance Analyzer") as demo:
        gr.Markdown(
            "# CLAP Maintenance Analyzer\n"
            "Upload or record machine audio to get CLAP-based issue detection, maintenance priority, and  diagnosis."
        )

        with gr.Row():
            audio_input = gr.Audio(
                type="filepath",
                label="Input Audio",
                sources=["upload", "microphone"],
            )
            with gr.Column():
                analyze_btn = gr.Button("Analyze Audio", variant="primary")
                clear_btn = gr.Button("Clear")

        with gr.Row():
            status_md = gr.Markdown(label="Maintenance Decision")
            with gr.Column():
                priority_box = gr.Textbox(label="Priority", interactive=False)
                top_signal_box = gr.Textbox(label="Top CLAP Signal", interactive=False)

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
        demo.launch()
