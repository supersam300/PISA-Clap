import sounddevice as sd
import soundfile as sf
import time
from datetime import datetime
import os


FS = 48000
DURATION = 5
CHANNELS = 1

def record_audio():
    print(f"Recording audio in {DURATION}-second intervals. Press Ctrl+C to stop.")

    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(os.path.dirname(__file__), f"recording_{timestamp}.wav")

        print(f"Recording started: {filename}")
        myrecording = sd.rec(int(DURATION * FS), samplerate=FS, channels=CHANNELS)

        for remaining in range(DURATION, 0, -1):
            bar_length = 30
            filled_length = int(bar_length * (DURATION - remaining) / DURATION)
            bar = '|' * filled_length + '-' * (bar_length - filled_length)
            print(f"Recording progress: [{bar}] {remaining}s remaining", end="\r")
            time.sleep(1)

        print(f"Recording progress: [{'|' * 30}] 0s remaining")
        sd.wait()

        sf.write(filename, myrecording, FS)
        print(f"Saved: {filename}")
        return filename
    
    except KeyboardInterrupt:
        print("\nRecording stopped by user.")



def delete_recordings():
    for files in os.listdir(os.path.dirname(__file__)):
        if files.endswith(".wav"):
            os.remove(os.path.join(os.path.dirname(__file__), files))

if __name__ == "__main__":
    record_audio()
