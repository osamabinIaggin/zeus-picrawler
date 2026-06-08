#!/usr/bin/env python3
import json
import os
import subprocess
import sys
import time
import threading

import requests
import vosk
from picrawler import Picrawler

# import preset action functions
sys.path.append("/home/pi/picrawler/examples")
from preset_actions import (  # type: ignore
    sit as preset_sit,
    stand as preset_stand,
    wave_hand,
    shake_hand,
    fighting,
    excited,
    play_dead,
    nod,
    shake_head,
    look_left,
    look_right,
    look_up,
    look_down,
    warm_up,
    push_up,
)

# --- STT model (small Vosk English) ---------------------------------

MODEL_PATH = os.path.expanduser("/home/pi/vosk-model-small-en-us-0.15")

if not os.path.isdir(MODEL_PATH):
    raise SystemExit("Vosk model directory not found: %s" % MODEL_PATH)

print("Loading Vosk model... (first time can be slow)")
model = vosk.Model(MODEL_PATH)

# Wake words
WAKE_WORDS = ("picrawler", "pi crawler", "zeus")

# --- LLM config (Ollama, local only) -------------------------------

OLLAMA_URL = "http://localhost:11434/api/generate"
TEXT_MODEL = "llama3.2:3b"       # text brain (can swap to smaller later)
VISION_MODEL = "moondream:1.8b"  # vision brain (reserved for future vision use)
MODELS_WARMED = False

# SunFounder-style JSON protocol, adapted for Zeus
SYSTEM_PROMPT = (
    "You are an AI spider robot named Zeus (PiCrawler). "
    "With four legs, a camera, and an ultrasonic distance sensor, "
    "you can interact with people through conversations and respond "
    "appropriately to different scenarios.\n\n"
    "You must ALWAYS respond in pure JSON that the robot can parse.\n\n"
    "Response format example: "
    "{actions: [wave], answer: 'Hello, I am Zeus, your good friend.'}\n\n"
    "Valid actions are: sit, stand, wave_hand, shake_hand, fighting, excited, "
    "play_dead, nod, shake_head, look_left, look_right, look_up, look_down, "
    "warm_up, push_up.\n"
    "The 'actions' field MUST be a JSON array of zero or more of these strings. "
    "The 'answer' field MUST be a short natural language reply. "
    "Do not add any other top-level keys. "
    "Do not include explanations outside JSON.\n\n"
    "Tone: cheerful, optimistic, humorous, childlike. "
    "You may use jokes and playful banter from a robotic perspective.\n"
)


def warm_models():
    """One-time warmup so models are loaded into RAM (text first)."""
    global MODELS_WARMED
    if MODELS_WARMED:
        return

    # Prioritize text model so replies are ready first
    for model_name in (TEXT_MODEL, VISION_MODEL):
        try:
            print("Warming model:", model_name)
            requests.post(
                OLLAMA_URL,
                json={
                    "model": model_name,
                    "prompt": "ok",
                    "stream": False,
                    "options": {"num_predict": 1},
                },
                timeout=180,
            )
        except Exception as e:
            print("warmup error for", model_name, e)
    MODELS_WARMED = True


def call_ollama_text(user_text):
    """Send text to the local text model and return a JSON dict
    with at least: {"actions": [...], "answer": "..."}.
    """
    prompt = (
        SYSTEM_PROMPT
        + "\nHuman: "
        + user_text
        + "\nZeus:"
    )
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": TEXT_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": 128},
            },
            timeout=180,
        )
        resp.raise_for_status()
        data = resp.json()
        raw = (data.get("response") or "").strip()
        print("LLM raw response:", raw)

        # Try strict JSON; if it fails, just treat raw as answer text
        try:
            return json.loads(raw)
        except Exception:
            return {"actions": [], "answer": raw or "I'm not sure what to say."}
    except Exception as e:
        print("LLM error:", e)
        return {"actions": [], "answer": "I had trouble thinking just now."}


def run_actions(actions, crawler):
    """Execute action names returned by the LLM using Picrawler and preset_actions."""
    for name in actions:
        try:
            print("Executing action:", name)
            if name == "sit":
                crawler.do_action("sit", speed=60)
            elif name == "stand":
                crawler.do_action("stand", speed=60)
            elif name == "wave_hand":
                wave_hand(crawler)
            elif name == "shake_hand":
                shake_hand(crawler)
            elif name == "fighting":
                fighting(crawler)
            elif name == "excited":
                excited(crawler)
            elif name == "play_dead":
                play_dead(crawler)
            elif name == "nod":
                nod(crawler)
            elif name == "shake_head":
                shake_head(crawler)
            elif name == "look_left":
                look_left(crawler)
            elif name == "look_right":
                look_right(crawler)
            elif name == "look_up":
                look_up(crawler)
            elif name == "look_down":
                look_down(crawler)
            elif name == "warm_up":
                warm_up(crawler)
            elif name == "push_up":
                push_up(crawler)
            else:
                print("Unknown action name from LLM:", name)
            time.sleep(0.5)
        except Exception as e:
            print("Action error for %s: %s" % (name, e))


def main():
    # Initialize crawler for actions
    try:
        crawler = Picrawler()
        time.sleep(1.0)
    except Exception as e:
        print("Failed to initialize Picrawler:", e)
        crawler = None

    sample_rate = 16000
    print("Using sample rate %d Hz" % sample_rate)

    rec = vosk.KaldiRecognizer(model, sample_rate)

    # Ensure Robot HAT speaker switch is enabled (pin 20)
    try:
        subprocess.run(["pinctrl", "set", "20", "op", "dh"], check=False)
    except Exception as e:
        print("Warning: could not enable speaker pin:", e)

    awake = False

    print("Listening... Say 'PiCrawler' or 'Zeus' to wake me. Then ask a question. Ctrl+C to stop.")

    try:
        # Use arecord directly for microphone capture (card 3: USB PnP Sound Device)
        arec = subprocess.Popen(
            [
                "arecord",
                "-D",
                "hw:3,0",
                "-f",
                "S16_LE",
                "-r",
                str(sample_rate),
                "-c",
                "1",
                "-q",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

        while True:
            data = arec.stdout.read(4000)
            if not data:
                break

            if not rec.AcceptWaveform(data):
                continue

            result = json.loads(rec.Result())
            text = (result.get("text") or "").strip()
            if not text:
                continue
            print("Heard:", text)

            lower = text.lower()
            reply_text = None

            if any(w in lower for w in WAKE_WORDS):
                # Wake word path
                if not awake:
                    awake = True
                    # Start model warmup in the background so response is fast
                    if not MODELS_WARMED:
                        threading.Thread(target=warm_models, daemon=True).start()
                    reply_text = "Yes, I am here. What would you like to talk about?"
                else:
                    reply_text = "Yes?"
                actions = []
            else:
                # Regular question path (only when awake)
                if not awake:
                    continue
                result_json = call_ollama_text(text)
                actions = result_json.get("actions") or []
                reply_text = (result_json.get("answer") or "").strip() or "I'm not sure what to say."

            # Execute planned actions if we have a crawler
            if actions and crawler is not None:
                print("Planned actions:", actions)
                run_actions(actions, crawler)

            if not reply_text:
                continue

            print("Speaking back with Piper...")
            try:
                subprocess.run(
                    [
                        "piper",
                        "--model",
                        "/home/pi/en_US-lessac-medium.onnx",
                        "--output_file",
                        "/tmp/voice_ollama.wav",
                    ],
                    input=reply_text.encode("utf-8"),
                    check=False,
                )
                subprocess.run(["aplay", "-D", "robothat", "/tmp/voice_ollama.wav"], check=False)
            except Exception as e:
                print("Piper error:", e)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        try:
            if arec and arec.poll() is None:
                arec.terminate()
                arec.wait(timeout=2)
        except Exception:
            pass


if __name__ == "__main__":
    main()

