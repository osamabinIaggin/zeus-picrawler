import json
import os
import subprocess

import pyaudio
import vosk

MODEL_PATH = os.path.expanduser("~/vosk-model-small-en-us-0.15")

if not os.path.isdir(MODEL_PATH):
    raise SystemExit(f"Vosk model directory not found: {MODEL_PATH}")

print("Loading Vosk model... (first time can be slow)")
model = vosk.Model(MODEL_PATH)

p = pyaudio.PyAudio()


WAKE_WORDS = ("picrawler", "pi crawler", "zeus")


def find_input_device_index():
    target_name = "USB PnP Sound Device"
    fallback = None
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        name = info.get("name", "")
        max_in = info.get("maxInputChannels", 0)
        if max_in > 0 and fallback is None:
            fallback = i
        if target_name in name and max_in > 0:
            print(f"Using input device {i}: {name}")
            return i
    if fallback is not None:
        info = p.get_device_info_by_index(fallback)
        print(f"Using fallback input device {fallback}: {info.get(name, )}")
    else:
        print("No input devices found; exiting.")
        raise SystemExit(1)
    return fallback


input_index = find_input_device_index()

# Fixed 44100 Hz to match USB mic default
sample_rate = 44100
print(f"Using sample rate {sample_rate} Hz")

rec = vosk.KaldiRecognizer(model, sample_rate)

# Ensure Robot HAT speaker switch is enabled (pin 20)
try:
    subprocess.run(["pinctrl", "set", "20", "op", "dh"], check=False)
except Exception as e:
    print(f"Warning: could not enable speaker pin: {e}")

stream = p.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=sample_rate,
    input=True,
    frames_per_buffer=4096,
    input_device_index=input_index,
)

awake = False

print("Listening... Say PiCrawler or Zeus to wake me. Ctrl+C to stop.")

try:
    while True:
        data = stream.read(4096, exception_on_overflow=False)
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            text = (result.get("text") or "").strip()
            if not text:
                continue
            print(f"Heard: {text}")

            lower = text.lower()
            reply = None

            if any(w in lower for w in WAKE_WORDS):
                if not awake:
                    awake = True
                    reply = "Yes, I am here."
                else:
                    reply = "Yes?"
            else:
                if not awake:
                    # Ignore non-wake speech when sleeping
                    continue
                # When awake, echo what was said (for now)
                reply = text

            if not reply:
                continue

            print("Speaking back with Piper...")
            try:
                proc = subprocess.run(
                    [
                        "piper",
                        "--model",
                        "/home/pi/en_US-lessac-medium.onnx",
                        "--output_file",
                        "/tmp/test_vosk_piper.wav",
                    ],
                    input=reply.encode("utf-8"),
                    check=False,
                )
                subprocess.run(["aplay", "-D", "robothat", "/tmp/test_vosk_piper.wav"], check=False)
            except Exception as e:
                print(f"Piper error: {e}")

except KeyboardInterrupt:
    print("\nStopping...")
finally:
    stream.stop_stream()
    stream.close()
    p.terminate()
