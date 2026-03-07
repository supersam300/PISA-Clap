import os
import sys
from audio_recs.record_audio import record_audio
from model.clap import clap_model
import time
import gradio as gr

AUDIO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "audio_recs"))

def gradio_clap(audio):
    return clap_model(audio=audio)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--loop", action="store_true", help="Run in continuous recording loop instead of Gradio")
    args = parser.parse_args()

    if args.loop:
        while True:
            try:
                record_audio()
                clap_model()

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
        print("Starting Gradio Interface. Use --loop to run continuous recording.")
        gr.Interface(fn=gradio_clap, inputs=gr.Audio(type="filepath"), outputs="text").launch()
