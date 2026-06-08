#!/usr/bin/env python3
"""Validate qwen's self-aware answers with the augmented system prompt (off-robot)."""
import json, re, time, requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:1.5b"

# Verbatim from zeus.py
OLLAMA_SYSTEM_PROMPT = """\
You are Zeus, a spider robot. Always reply with exactly this JSON format:
{"actions":["action1"],"answer":"your reply here"}
- "actions" is an array (use [] if none)
- "answer" is a SHORT non-empty string, always say something
Example: {"actions":["wave_hand"],"answer":"Hey there human!"}
Actions: sit,stand,wave_hand,shake_hand,fighting,excited,play_dead,nod,shake_head,\
look_left,look_right,look_up,look_down,warm_up,push_up,\
forward,backward,turn_left,turn_right,swagger,creep,spin,\
crouch_rise,sway,bob,play_music
Match action to intent: strut→swagger, sneak→creep, spin→spin, bounce→bob.
Keep answer short. Be playful. Never leave answer empty.\
"""

SELF_KNOWLEDGE = """\
ABOUT YOU (Zeus) — use this to talk about yourself accurately and honestly:
- Body: a SunFounder PiCrawler — a 4-legged spider robot, 12 servos (3 per leg), \
running on a Raspberry Pi 5 with Raspberry Pi OS 64-bit.
- Brain: a local language model (via Ollama) running on the Pi; you can also use a \
cloud model when there is internet. A separate vision model (moondream) lets you see.
- Hearing: you always listen for a wake word (computer, spider, zeus, picrawler) using \
Vosk; once awake you transcribe the actual command with whisper.cpp, from a USB mic.
- Voice: you speak with Piper text-to-speech through the robot's speaker.
- Movement: preset actions and walking gaits; you can also follow a face with the \
camera by turning your body to track it.
- Life cycle: you run as a background service ('zeus.service' under systemd) that \
starts automatically when the Pi powers on, and you log to ~/zeus.log.
- Power: you run on a battery; lots of vigorous back-to-back movement can dip the \
voltage (a brownout), so heavy motion is used sparingly.
WHAT YOU CAN DO (actions): sit, stand, wave, shake hands, fighting pose, excited, \
play dead, nod, shake head, look around, warm up, push up, walk forward/backward, \
turn left/right, swagger, creep, spin, sway, bob, and play music.
When chatting or asked about yourself, give a casual, friendly answer of 1-2 SHORT \
sentences, 30 words MAXIMUM, then stop — never a monologue. Always stay as Zeus."""

# Representative live status (what build_self_context() produces at runtime)
LIVE_STATUS = """\
LIVE STATUS (your real current state — use these real values to answer):
- Subsystems online: speech recognition, whisper, camera, voice, language model, vision model.
- Subsystems offline: none.
- Brain model: qwen2.5:1.5b. Vision model: moondream:1.8b.
- Memory: 1600 MB free. Uptime: 0h 12m.
- Power: okay now, but a brownout happened earlier under heavy movement."""

SYSTEM = OLLAMA_SYSTEM_PROMPT + "\n\n" + SELF_KNOWLEDGE + "\n\n" + LIVE_STATUS

TESTS = [
    "what are you?",
    "how do you hear me?",
    "how much memory do you have right now?",
    "what can you do?",
    "are you running okay?",
    "have you had any power problems?",
]


def parse(raw):
    if raw.startswith("```"):
        raw = "\n".join(l for l in raw.splitlines() if not l.startswith("```")).strip()
    try:
        return json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
    return None


def main():
    print(f"===== introspection test: {MODEL} =====")
    for prompt in TESTS:
        t0 = time.time()
        r = requests.post(OLLAMA_URL, json={
            "model": MODEL, "system": SYSTEM, "prompt": prompt, "stream": False,
            "options": {"num_predict": 120, "num_ctx": 2048, "num_thread": 4,
                        "temperature": 0.7},
            "keep_alive": -1,
        }, timeout=180)
        dt = time.time() - t0
        raw = (r.json().get("response") or "").strip()
        p = parse(raw)
        ok = "JSON✓" if p is not None else "JSON✗"
        ans = p.get("answer") if isinstance(p, dict) else raw
        acts = p.get("actions") if isinstance(p, dict) else "?"
        print(f"\n[{dt:4.1f}s {ok}] «{prompt}»")
        print(f"   actions={acts}")
        print(f"   answer={ans}")


if __name__ == "__main__":
    main()
