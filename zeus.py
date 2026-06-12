#!/usr/bin/env python3
"""
Zeus — Autonomous AI Spider Robot
Combines: face tracking + wake word + Vosk STT + Ollama LLM + Piper TTS + preset actions

Latency improvements (vs original):
- Internet status cached 60s (saves ~236 ms/call)
- OpenAI timeout tightened 15s → 10s
- OpenAI 429 rate-limit handled explicitly (instant fallback, no retry delay)
- System prompt shortened ~40% (saves ~80 Ollama prompt tokens → ~3 s faster)
- Ollama num_predict 130 → 60 (answer fits in ~30 tokens; stops wasted generation)
- Ollama keep_alive=-1 (model stays in RAM between calls, removes ~350 ms load)
- Ollama uses native system/prompt fields for cleaner KV-cache reuse
"""

import array
import audioop
import base64
import glob
import json
import math
import os
import random
import re
import subprocess
import sys
import time
import threading
import wave

import cv2
import requests
import vosk
from picrawler import Picrawler
from robot_hat import Servo
from vilib import Vilib

sys.path.append("/home/pi/picrawler/examples")
from preset_actions import (  # type: ignore
    wave_hand, shake_hand, fighting, excited,
    play_dead, nod, shake_head, look_left, look_right,
    look_up, look_down, warm_up, push_up,
)


# ── Robot init ────────────────────────────────────────────────────────────────
# Brownout mitigation: the Pi 5 and the 12 servos share the battery. At boot,
# CPU load + servo inrush together sag the rail (confirmed under-voltage,
# vcgencmd throttled=0x50000 on battery). We cap the CPU during the servo-init
# window so the battery has headroom for the servos, then restore it.
def _read_cpu_governor():
    try:
        with open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor") as f:
            return f.read().strip()
    except Exception:
        return "ondemand"

def _set_cpu_governor(gov):
    paths = glob.glob("/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor")
    ok = False
    for p in paths:
        try:
            with open(p, "w") as f:
                f.write(gov)
            ok = True
        except Exception:
            pass
    if ok:
        print(f"[power] CPU governor → {gov}")
    else:
        print(f"[power] could not set CPU governor → {gov} (needs root?)")

_orig_governor = _read_cpu_governor()
_set_cpu_governor("powersave")   # cut Pi current draw during servo inrush

# SDK energises all 12 servos one-by-one (staggered) to avoid a current spike.
print("Initializing PiCrawler...")
crawler = Picrawler()
time.sleep(1.5)                  # let the rail settle after MCU reset + servo energise

# Stand slowly: low speed = small per-step angle deltas = lower peak current.
# (The old code also wiggled all 12 servos to 10°/0° first — removed; the SDK
# already zeroes them one-by-one, so the wiggle was redundant startup load.)
print("Standing up (gentle)...")
crawler.do_action("stand", speed=20)
time.sleep(1.0)

_set_cpu_governor(_orig_governor)   # restore normal CPU scaling once settled

# ── Parallel init: camera, speaker, Vosk all start concurrently ──────────────
subprocess.run(["pinctrl", "set", "20", "op", "dh"], check=False)

# Prefer the larger lgraph model (much better on accented speech) when present,
# fall back to the small model so STT never hard-fails.
_VOSK_CANDIDATES = [
    "/home/pi/vosk-model-en-us-0.22-lgraph",
    "/home/pi/vosk-model-small-en-us-0.15",
]
MODEL_PATH = next((p for p in _VOSK_CANDIDATES if os.path.isdir(p)), None)
if MODEL_PATH is None:
    raise SystemExit("No Vosk model found in: " + ", ".join(_VOSK_CANDIDATES))
print(f"[stt] using Vosk model: {MODEL_PATH}")

_camera_ready = threading.Event()
_vosk_ready   = threading.Event()
vosk_model    = None

def _init_camera():
    print("Starting camera...")
    Vilib.camera_start(vflip=False, hflip=False)
    Vilib.display(local=False, web=True)
    Vilib.face_detect_switch(True)
    _camera_ready.set()
    print("Camera ready")

def _init_vosk():
    global vosk_model
    print("Loading Vosk model...")
    vosk_model = vosk.Model(MODEL_PATH)
    _vosk_ready.set()
    print("Vosk ready")

threading.Thread(target=_init_camera, daemon=True).start()
threading.Thread(target=_init_vosk, daemon=True).start()

# ── Face tracking constants ───────────────────────────────────────────────────
STAND_Z     = [-60, -60, -60, -60]
LOOK_UP_Z   = [-86, -86, -48, -40]
LOOK_DOWN_Z = [-38, -50, -78, -86]
SEARCH_SMOOTHING = 0.15
# Adaptive tracking — alpha scales up with error for fast catch-up, low at idle
TRACK_ALPHA_MIN = 0.25    # smooth idle when face is centered
TRACK_ALPHA_MAX = 0.70    # fast catch-up when face is far off-centre
YAW_RANGE       = 25      # max yaw degrees (±) — was 20
YAW_DEAD_PX     = 15      # horizontal pixel dead zone — ignore tiny drift
PITCH_DEAD_PX   = 20      # vertical pixel dead zone — ignore tiny drift
TRACK_SPEED     = 60      # servo speed for tracking moves — gentler than 100 to
                          # cut peak current (the old fast snap spiked the rail)

# Shared state for face tracking
_current_yaw    = 0.0
_current_pitch_t = 0.0

# ── Interaction state ─────────────────────────────────────────────────────────
# Face tracking owns the body, but it must yield whenever the robot is attending
# to the user — otherwise it keeps moving the legs (and making servo noise into
# the mic) while we speak or record a command. These named events make the
# coordination explicit instead of being tangled across the call sites.
action_in_progress = threading.Event()   # an action/gait is executing
is_speaking        = threading.Event()   # TTS is playing (also suppresses mic)
is_listening       = threading.Event()   # recording a command for whisper
zeus_sleeping      = threading.Event()   # idle/asleep until wake word

def tracking_should_pause():
    """Face tracking yields the body while asleep, acting, speaking, or listening."""
    return (zeus_sleeping.is_set() or action_in_progress.is_set()
            or is_speaking.is_set() or is_listening.is_set())

# ── Movement safety gate ──────────────────────────────────────────────────────
# When ~/.zeus_no_move exists, all servo motion (face tracking, idle search,
# first-wake wave, LLM actions) is suppressed while the FULL voice pipeline
# still runs. Use this on weak power to test speech without browning out the Pi;
# delete the flag (or fit a stronger battery) to re-enable movement.
_NO_MOVE_FLAG    = "/home/pi/.zeus_no_move"     # suppress ALL servo motion
_NO_TRACK_FLAG   = "/home/pi/.zeus_no_track"    # suppress only continuous tracking
MOVEMENT_ENABLED = not os.path.exists(_NO_MOVE_FLAG)
# Continuous face tracking + idle search keep the servos under near-constant load,
# which browns out the Pi on weak power. They can be disabled on their own while
# still allowing brief, discrete moves (first-wake wave, LLM-requested actions).
FACE_TRACK_ENABLED = MOVEMENT_ENABLED and not os.path.exists(_NO_TRACK_FLAG)
print(f"[movement] enabled={MOVEMENT_ENABLED} face_tracking={FACE_TRACK_ENABLED}")

_last_face_time    = 0.0   # epoch; updated by face_track_thread
_last_command_time = 0.0   # epoch; updated by voice loop

# Signals the voice loop to flush and restart arecord (clears stale buffer after sleep)
_reset_mic = threading.Event()
# While set, the voice loop releases the mic so the web intercom can record.
call_mode = threading.Event()


def lerp(a, b, t):
    return a + (b - a) * t


def get_pitch_z(target_z, t):
    return [lerp(STAND_Z[i], target_z[i], t) for i in range(4)]


def apply_pose(yaw, pitch_z, speed=100):
    step = [
        [45 - yaw * 0.3, 45 + yaw, pitch_z[0]],  # right front
        [45 + yaw * 0.3, 45 - yaw, pitch_z[1]],  # left front
        [45 - yaw * 0.3, 45 + yaw, pitch_z[2]],  # left rear
        [45 + yaw * 0.3, 45 - yaw, pitch_z[3]],  # right rear
    ]
    crawler.do_step(step, speed=speed)


def face_track_thread():
    global _current_yaw, _current_pitch_t, _last_face_time

    last_face_time = time.time()
    searching     = False
    search_dir_h  = 1
    search_h      = 0.0
    search_phase  = 0
    pitch_t       = 0.0

    while True:
        # Yield the body while asleep, acting, speaking, or listening to a command
        if tracking_should_pause():
            time.sleep(0.1)
            continue

        fn = Vilib.detect_obj_parameter.get("human_n", 0)

        if fn > 0:
            fx = Vilib.detect_obj_parameter.get("human_x", 320)
            fy = Vilib.detect_obj_parameter.get("human_y", 240)

            x_err = fx - 320
            y_err = fy - 240

            # Dead zone: don't chase sub-threshold offsets (prevents jitter)
            target_yaw   = (x_err / 320) * YAW_RANGE if abs(x_err) > YAW_DEAD_PX  else _current_yaw
            target_pitch = (y_err / 240)              if abs(y_err) > PITCH_DEAD_PX else _current_pitch_t

            # Adaptive alpha: large error → fast catch-up; near centre → smooth
            yaw_alpha   = min(TRACK_ALPHA_MAX, TRACK_ALPHA_MIN + abs(target_yaw   - _current_yaw)   * 0.025)
            pitch_alpha = min(TRACK_ALPHA_MAX, TRACK_ALPHA_MIN + abs(target_pitch - _current_pitch_t) * 0.40)

            _current_yaw     = lerp(_current_yaw, target_yaw, yaw_alpha)
            _current_pitch_t = lerp(_current_pitch_t, target_pitch, pitch_alpha)

            if _current_pitch_t < 0:
                pitch_z = get_pitch_z(LOOK_UP_Z, min(abs(_current_pitch_t), 1.0))
            else:
                pitch_z = get_pitch_z(LOOK_DOWN_Z, min(_current_pitch_t, 1.0))

            apply_pose(_current_yaw, pitch_z, speed=TRACK_SPEED)

            last_face_time = time.time()
            _last_face_time = last_face_time   # update global for sleep monitor
            searching    = False
            search_h     = 0.0
            search_phase = 0
            pitch_t      = 0.0

        else:
            elapsed = time.time() - last_face_time

            if elapsed > 3:
                if not searching:
                    searching    = True
                    search_phase = 0
                    search_h     = 0.0
                    search_dir_h = 1
                    pitch_t      = 0.0

                if search_phase == 0:
                    target_h = search_h + search_dir_h * 2
                    if target_h > 20:
                        search_dir_h = -1
                    elif target_h < -20:
                        search_dir_h = 1
                        search_phase = 1
                    search_h = target_h
                    pitch_t  = lerp(pitch_t, 0, SEARCH_SMOOTHING)
                    pitch_z  = (STAND_Z[:]
                                if abs(pitch_t) < 0.01
                                else get_pitch_z(LOOK_UP_Z if pitch_t < 0 else LOOK_DOWN_Z, abs(pitch_t)))

                elif search_phase == 1:
                    target_h = search_h + search_dir_h * 2
                    if target_h > 20:
                        search_dir_h = -1
                    elif target_h < -20:
                        search_dir_h = 1
                        search_phase = 2
                    search_h = target_h
                    pitch_t  = lerp(pitch_t, -1.0, 0.03)
                    pitch_z  = get_pitch_z(LOOK_UP_Z, min(abs(pitch_t), 1.0))

                elif search_phase == 2:
                    target_h = search_h + search_dir_h * 2
                    if target_h > 20:
                        search_dir_h = -1
                    elif target_h < -20:
                        search_dir_h = 1
                        search_phase = 3
                    search_h = target_h
                    pitch_t  = lerp(pitch_t, 1.0, 0.03)
                    pitch_z  = get_pitch_z(LOOK_UP_Z if pitch_t < 0 else LOOK_DOWN_Z, abs(pitch_t))

                else:
                    pitch_t  = lerp(pitch_t, 0, SEARCH_SMOOTHING)
                    search_h = lerp(search_h, 0, SEARCH_SMOOTHING)
                    if abs(pitch_t) < 0.05 and abs(search_h) < 1:
                        pitch_t      = 0.0
                        search_h     = 0.0
                        search_phase = 0
                        pitch_z      = STAND_Z[:]
                    else:
                        pitch_z = get_pitch_z(
                            LOOK_UP_Z if pitch_t < 0 else LOOK_DOWN_Z, abs(pitch_t))

                _current_yaw     = lerp(_current_yaw, search_h, SEARCH_SMOOTHING)
                _current_pitch_t = pitch_t
                apply_pose(_current_yaw, pitch_z, speed=TRACK_SPEED)

        time.sleep(0.04)


# Wake words are detected by the lightweight always-on Vosk recognizer.
# "computer" and "spider" tested as the most reliably recognized (incl. accent);
# "zeus"/"picrawler" kept as bonus triggers.
WAKE_WORDS  = ("computer", "spider", "zeus", "picrawler", "pi crawler")

# ── Shortcuts (instant response, no LLM call) ─────────────────────────────────
# Keys are matched as substrings in order of length (longest first).
# Value: (actions_list, spoken_reply)
SHORTCUTS = {
    # Greetings / social
    "say hi":        (["wave_hand"],  "Hey there!"),
    "wave":          (["wave_hand"],  "Hey! How's it going?"),
    "say hello":     (["wave_hand"],  "Hello! I'm Zeus, your friendly spider robot!"),
    # Moves
    "come forward":  (["forward"],    "Coming your way!"),
    "move forward":  (["forward"],    "Moving forward!"),
    "go forward":    (["forward"],    "On my way!"),
    "come here":     (["forward"],    "Coming!"),
    "go back":       (["backward"],   "Backing up!"),
    "back up":       (["backward"],   "Going back!"),
    "move back":     (["backward"],   "Moving back!"),
    "turn left":     (["turn_left"],  "Turning left!"),
    "turn right":    (["turn_right"], "Turning right!"),
    # Tricks
    "push up":       (["push_up"],    "Let's get those gains!"),
    "pushup":        (["push_up"],    "Here we go!"),
    "warm up":       (["warm_up"],    "Stretching it out!"),
    "play dead":     (["play_dead"],  "Bleh... I'm dead."),
    "fight":         (["fighting"],   "Put up your fists!"),
    "dance":         (["excited"],    "Let's go!"),
    # Posture
    "sit down":      (["sit"],        "Sitting down."),
    "stand up":      (["stand"],      "Standing up!"),
    "nod":           (["nod"],        "Yes!"),
    "shake your head": (["shake_head"],  "Nope!"),
    # style
    "swagger":         (["swagger"],     "Watch me strut!"),
    "strut":           (["swagger"],     "Too cool for school."),
    "creep":           (["creep"],       "Sneaking..."),
    "sneak":           (["creep"],       "You won't even hear me."),
    "spin":            (["spin"],        "Wheeeee!"),
    "spin around":     (["spin"],        "Getting dizzy!"),
    # body
    "crouch":          (["crouch_rise"], "Getting low!"),
    "sway":            (["sway"],        "Side to side!"),
    "bob":             (["bob"],         "Boing boing!"),
    "bounce":          (["bob"],         "Boing!"),
}


def match_shortcut(text):
    """Return (actions, reply) if text matches a shortcut phrase, else None."""
    lower = text.lower()
    for key in sorted(SHORTCUTS, key=len, reverse=True):  # longest match first
        if key in lower:
            return SHORTCUTS[key]
    return None


# ── LLM — OpenAI (online) + Ollama (offline fallback) ────────────────────────
OLLAMA_URL   = "http://localhost:11434/api/generate"
TEXT_MODEL   = "qwen2.5:0.5b"   # 3x faster than 1.5b, still holds the JSON contract
VISION_MODEL = "moondream:1.8b"
MODELS_WARMED = False

def _load_env_file(path):
    """Minimal .env reader (no python-dotenv dependency). KEY=VALUE per line."""
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    except FileNotFoundError:
        pass


# Secrets live in ~/.env (or alongside this script), never hardcoded in source.
for _envp in (os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"),
              os.path.expanduser("~/.env")):
    _load_env_file(_envp)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_URL     = "https://api.openai.com/v1/chat/completions"
OPENAI_MODEL   = "gpt-4o-mini"
if not OPENAI_API_KEY:
    print("[config] no OPENAI_API_KEY in env — cloud LLM/vision disabled, using local only")

# ── Internet status cache ─────────────────────────────────────────────────────
# FIX: was checking internet on every call (~236 ms overhead).
# Now cached for 60 seconds.
_internet_cache: bool | None = None
_internet_cache_time: float = 0.0
_INTERNET_CACHE_TTL = 60.0  # seconds


def has_internet() -> bool:
    """Check internet, caching the result for 60 seconds."""
    global _internet_cache, _internet_cache_time
    now = time.monotonic()
    if _internet_cache is not None and (now - _internet_cache_time) < _INTERNET_CACHE_TTL:
        return _internet_cache
    try:
        requests.head("https://api.openai.com", timeout=3)
        result = True
    except Exception:
        result = False
    _internet_cache = result
    _internet_cache_time = now
    print(f"[internet] checked: {result} (cached for {_INTERNET_CACHE_TTL:.0f}s)")
    return result


def invalidate_internet_cache():
    """Call after OpenAI errors to force a recheck next time."""
    global _internet_cache
    _internet_cache = None


# ── System prompts ────────────────────────────────────────────────────────────
# FIX: original prompt was ~215 tokens.  Ollama eval costs ~600 ms/token on
# Pi 5 with llama3.2:3b, so every extra prompt token adds latency.
# Stripped to ~130 tokens while preserving all required semantics.

# Full prompt for OpenAI (network latency dominates, token count doesn't matter)
OPENAI_SYSTEM_PROMPT = """\
You are Zeus, an AI spider robot (PiCrawler). Four legs, camera, ultrasonic sensor.
Respond ONLY with a single valid JSON object — no markdown, no extra text.
Format: {"actions": [...], "answer": "..."}

Valid actions:
- Expressions: sit, stand, wave_hand, shake_hand, fighting, excited, play_dead, nod, shake_head, look_left, look_right, look_up, look_down, warm_up, push_up
- Walking: forward, backward, turn_left, turn_right
- Style: swagger (slow cool walk), creep (ultra-slow stalk), spin (360 spin)
- Body: crouch_rise (dramatic crouch+rise), sway (side-to-side), bob (bounce)
- Fun: play_music

Pick actions that fit the mood. Chain multiple when it makes sense.
Examples: strut→swagger, sneak→creep, dizzy→spin, bounce→bob,
show off→[excited,swagger], celebrate→[spin,play_music,excited]
"actions" can be []. "answer" is short and punchy. Be witty.\
"""

# Shorter prompt for Ollama — every token saved = ~600 ms faster on Pi 5
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


# ── Self-knowledge: lets Zeus talk accurately about its own design + state ──────
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


def _meminfo_available_mb():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
    except Exception:
        pass
    return None


def _uptime_str():
    try:
        with open("/proc/uptime") as f:
            secs = float(f.read().split()[0])
        h, rem = divmod(int(secs), 3600)
        return f"{h}h {rem // 60}m"
    except Exception:
        return "unknown"


def build_self_context():
    """Live snapshot of Zeus's own state, injected so it can answer about itself."""
    try:
        ok, fail = run_self_check()
    except Exception:
        ok, fail = [], []
    mem = _meminfo_available_mb()
    mem_str = f"{mem} MB free" if mem is not None else "unknown"
    try:
        out = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True,
                             text=True, timeout=3).stdout.strip()
        val = int(out.split("=")[1], 16) if "=" in out else 0
        if val & 0x1:
            power = "under-voltage RIGHT NOW (battery is low)"
        elif val & 0x10000:
            power = "okay now, but a brownout happened earlier under heavy movement"
        else:
            power = "healthy"
    except Exception:
        power = "unknown"
    return SELF_KNOWLEDGE + "\n\n" + (
        "LIVE STATUS (your real current state — use these real values to answer):\n"
        f"- Subsystems online: {', '.join(ok) if ok else 'none'}.\n"
        f"- Subsystems offline: {', '.join(fail) if fail else 'none'}.\n"
        f"- Brain model: {TEXT_MODEL}. Vision model: {VISION_MODEL}.\n"
        f"- Memory: {mem_str}. Uptime: {_uptime_str()}.\n"
        f"- Power: {power}.\n"
        f"- Right now it's {time_of_day()} and your mood is {current_mood()} "
        f"(let it color your tone, subtly)."
    )


# Questions about Zeus itself (or casual chat) route to the conversational path.
_SELF_QUERY_RE = re.compile(
    r"\b(your|you're|yourself|who are you|what are you|how are you|how do you|"
    r"how does your|what can you do|what do you do|your name|your brain|your model|"
    r"your memory|your ram|your battery|your power|your code|your software|"
    r"your hardware|your servo|your camera|your mic|your voice|your sensor|"
    r"your system|your service|your process|are you (ok|okay|alive|running|working)|"
    r"tell me about yourself|feeling|your status|self.?check|diagnostic|how you work|"
    r"do you have|do you run|running on|what model|have you (had|been|ever)|"
    r"are you (ok|okay|alive|running|working|broken)|been (ok|okay|alright)|"
    r"any (power|brownout|battery|memory|cpu|servo|sensor|hardware|software) "
    r"(problem|issue|trouble)|power problem|brownout|"
    r"how much (memory|ram|battery|power|space|storage|cpu|disk))\b",
    re.IGNORECASE,
)


def is_self_query(text):
    return bool(_SELF_QUERY_RE.search(text or ""))


def warm_models():
    global MODELS_WARMED
    if MODELS_WARMED:
        return
    try:
        print("Warming:", TEXT_MODEL)
        requests.post(OLLAMA_URL, json={
            "model": TEXT_MODEL,
            "system": OLLAMA_SYSTEM_PROMPT,
            "prompt": "hi",
            "stream": False,
            "options": {"num_predict": 1, "num_ctx": 2048, "num_thread": 4},
            "keep_alive": -1,
        }, timeout=180)
    except Exception as e:
        print("Warmup error:", TEXT_MODEL, e)
    try:
        print("Warming:", VISION_MODEL, "(with image — loads the vision tower)")
        # Warm the VISION path, not just text: a text-only prompt leaves the image
        # projector cold, making the first real query ~108s instead of ~14s. Send a
        # tiny dummy image so the encoder is resident before the user asks.
        import numpy as np  # noqa: local import; only needed here
        ok, buf = cv2.imencode(".jpg", np.zeros((32, 32, 3), dtype=np.uint8))
        dummy_b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        requests.post(OLLAMA_URL, json={
            "model": VISION_MODEL, "prompt": "ok", "images": [dummy_b64],
            "stream": False, "options": {"num_predict": 1},
            "keep_alive": -1,
        }, timeout=180)
    except Exception as e:
        print("Warmup error:", VISION_MODEL, e)
    MODELS_WARMED = True


def _parse_llm_raw(raw):
    """Parse LLM output as JSON, with fallbacks for truncated/malformed output."""
    # Strip code fences if model wraps in ```json ... ```
    if raw.startswith("```"):
        raw = "\n".join(l for l in raw.splitlines() if not l.startswith("```")).strip()

    # 1. Try strict JSON
    try:
        return json.loads(raw)
    except Exception:
        pass

    # 2. Find outermost { } block (handles leading/trailing junk)
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass

    # 3. Truncated JSON — extract fields individually via regex
    answer_m  = re.search(r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)', raw)
    actions_m = re.search(r'"actions"\s*:\s*\[(.*?)\]', raw, re.DOTALL)
    actions = []
    if actions_m:
        try:
            actions = json.loads("[" + actions_m.group(1) + "]")
        except Exception:
            pass
    if answer_m:
        return {"actions": actions, "answer": answer_m.group(1).rstrip('"\\')}

    # 4. Give up — return raw text so at least something is said
    return {"actions": [], "answer": raw or "I'm not sure what to say."}


def call_llm(user_text):
    if OPENAI_API_KEY and has_internet():
        result = _call_openai(user_text)
        if result is not None:
            return result
        # OpenAI failed (rate limit / network); force recheck internet next time
        invalidate_internet_cache()
    print("Using local Ollama")
    return _call_ollama(user_text)


def _call_openai(user_text):
    """Returns parsed dict on success, or None on any error (caller falls back to Ollama)."""
    try:
        resp = requests.post(OPENAI_URL, json={
            "model": OPENAI_MODEL,
            "messages": [
                {"role": "system", "content": OPENAI_SYSTEM_PROMPT},
                {"role": "user",   "content": user_text},
            ],
            "max_tokens": 100,   # short JSON reply never needs more than ~60 tokens
        }, headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }, timeout=10)          # FIX: was 15s; 10s is plenty, fail fast

        # Handle 429 rate-limit explicitly — fall through to Ollama immediately
        if resp.status_code == 429:
            print("OpenAI 429 rate-limited — falling back to Ollama")
            return None

        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        print("OpenAI:", raw)
        return _parse_llm_raw(raw)
    except requests.exceptions.Timeout:
        print("OpenAI timeout — falling back to Ollama")
        return None
    except Exception as e:
        print("OpenAI error:", e, "— falling back to Ollama")
        return None


def _call_ollama(user_text):
    try:
        resp = requests.post(OLLAMA_URL, json={
            "model":  TEXT_MODEL,
            "system": OLLAMA_SYSTEM_PROMPT,
            "prompt": user_text,
            "stream": False,
            "options": {
                "num_predict": 80,
                "num_ctx": 2048,
                "num_thread": 4,
                "temperature": 0.7,
            },
            "keep_alive": -1,
        }, timeout=180)
        resp.raise_for_status()
        raw = (resp.json().get("response") or "").strip()
        print("Ollama raw:", raw)
        parsed = _parse_llm_raw(raw)
        # Guard against model returning empty/null answer
        if not (parsed.get("answer") or "").strip():
            parsed["answer"] = "Beep boop, I'm thinking!"
        return parsed
    except Exception as e:
        print("Ollama error:", e)
        return {"actions": [], "answer": "I had trouble thinking just now."}


_SENTENCE_END_RE = re.compile(r'[.!?](?:\s|"|$)')


def _stream_ollama_and_respond(user_text, system=None, num_predict=80):
    """Stream Ollama tokens, execute actions as soon as parsed, speak sentences as they complete.

    `system` overrides the default command prompt (used for the self-aware chat path);
    `num_predict` caps answer length (higher for conversation, lower for quick commands).
    """
    if system is None:
        system = OLLAMA_SYSTEM_PROMPT
    action_in_progress.set()
    is_speaking.set()
    try:
        resp = requests.post(OLLAMA_URL, json={
            "model": TEXT_MODEL,
            "system": system,
            "prompt": user_text,
            "stream": True,
            "options": {"num_predict": num_predict, "num_ctx": 2048, "num_thread": 4, "temperature": 0.7},
            "keep_alive": -1,
        }, timeout=180, stream=True)
        resp.raise_for_status()

        raw = ""
        actions_done = False
        in_answer = False
        answer_buf = ""
        spoken_len = 0

        for line in resp.iter_lines():
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue
            token = chunk.get("response", "")
            raw += token

            if not actions_done:
                m = re.search(r'"actions"\s*:\s*\[([^\]]*)\]', raw)
                if m:
                    actions_done = True
                    try:
                        actions = json.loads("[" + m.group(1) + "]")
                    except Exception:
                        actions = []
                    run_actions(actions)

            if not in_answer:
                m2 = re.search(r'"answer"\s*:\s*"', raw)
                if m2:
                    in_answer = True
                    answer_buf = raw[m2.end():]
            elif token:
                answer_buf += token

            if in_answer:
                clean = answer_buf.split('"')[0]
                unsent = clean[spoken_len:]
                sm = _SENTENCE_END_RE.search(unsent)
                if sm and len(unsent) > sm.end():
                    sentence = unsent[:sm.end()].strip()
                    if sentence:
                        speak(sentence)
                        spoken_len += sm.end()

            if chunk.get("done"):
                break

        if in_answer:
            clean = answer_buf.split('"')[0].strip()
            remaining = clean[spoken_len:].strip()
            if remaining:
                speak(remaining)
            elif spoken_len == 0:
                parsed = _parse_llm_raw(raw)
                speak((parsed.get("answer") or "").strip() or "I'm not sure what to say.")
        else:
            parsed = _parse_llm_raw(raw)
            if not actions_done:
                run_actions(parsed.get("actions") or [])
            speak((parsed.get("answer") or "").strip() or "I'm not sure what to say.")

        print("Ollama streamed:", raw[:200])

    except Exception as e:
        print("Streaming Ollama error:", e)
        speak("I had trouble thinking just now.")
    finally:
        action_in_progress.clear()
        is_speaking.clear()


def _answer_self_query(text):
    """Conversational/introspective answer with live self-context injected.

    Stays local (the answer is about THIS machine) and gives a longer answer budget.
    """
    system = OLLAMA_SYSTEM_PROMPT + "\n\n" + build_self_context()
    _stream_ollama_and_respond(text, system=system, num_predict=120)


# ── Phase 5: vision — let Zeus actually SEE via the moondream model ────────────
# moondream:1.8b is already resident in Ollama (warmed at boot). We grab the
# current camera frame (Vilib.flask_img), JPEG-encode it, and ask moondream the
# user's question. Plain prose back (moondream isn't a JSON/persona model), spoken
# as-is with a light lead-in. Works in no-move mode — Zeus sees whatever's in front.
_VISION_QUERY_RE = re.compile(
    r"\b(what (do|can) you see|what'?s (in front|out there|that|this)|"
    r"look at (this|that|it|me|my)|can you see|do you see|what am i holding|"
    r"what'?s in your (view|sight)|describe (what|this|that|the (scene|view|room))|"
    r"take a (photo|picture|look)|use your (eyes|camera|vision)|"
    r"what does it look like|what is (this|that)|read (this|the)|"
    r"who (is|am i)|how many (people|fingers)|what colou?r)\b",
    re.IGNORECASE,
)

_VISION_LEADINS = ["Let me look... ", "Looking... ", "Hmm, I see... ", "Okay... "]
_VISION_PROMPT_DEFAULT = "Describe what you see in one or two short sentences."
_VISION_MAX_W          = 384   # downscale width — smaller upload + lighter moondream
_VISION_TIMEOUT_CLOUD  = 15
_VISION_TIMEOUT_LOCAL  = 90    # CPU inference on the 1.8b model is slow (~14s warm)
# Spoken while moondream (offline) grinds, so the user isn't left in silence.
_VISION_LOCAL_WAIT = ["Let me take a good look, one sec.",
                      "Hold on, focusing my eyes.",
                      "Give me a moment to really look."]


def is_vision_query(text):
    return bool(_VISION_QUERY_RE.search(text or ""))


def _capture_jpeg_b64():
    """Grab the current camera frame, downscaled, as base64 JPEG (or None)."""
    try:
        frame = Vilib.flask_img            # current frame (numpy/cv2 image)
        h, w = frame.shape[:2]
        if w > _VISION_MAX_W:
            frame = cv2.resize(frame, (_VISION_MAX_W, int(h * _VISION_MAX_W / w)))
        ok, buf = cv2.imencode(".jpg", frame)
        if not ok:
            return None
        return base64.b64encode(buf.tobytes()).decode("ascii")
    except Exception as e:
        print("[vision] frame capture failed:", e)
        return None


def _openai_vision(question, img_b64):
    """Ask gpt-4o-mini about the image. Returns description str, or None on failure."""
    try:
        resp = requests.post(OPENAI_URL, json={
            "model": OPENAI_MODEL,
            "messages": [
                {"role": "system", "content":
                 "You are Zeus, a witty spider robot describing what your camera sees. "
                 "Answer in one or two short, natural spoken sentences."},
                {"role": "user", "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                ]},
            ],
            "max_tokens": 120,
        }, headers={"Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json"}, timeout=_VISION_TIMEOUT_CLOUD)
        if resp.status_code == 429:
            print("[vision] OpenAI 429 — falling back to moondream")
            return None
        resp.raise_for_status()
        desc = resp.json()["choices"][0]["message"]["content"].strip()
        print("[vision] gpt-4o-mini:", repr(desc))
        return desc or None
    except Exception as e:
        print("[vision] OpenAI vision error:", e, "— falling back to moondream")
        return None


def _moondream_vision(question, img_b64):
    """Ask local moondream about the image. Returns description str, or '' on failure."""
    try:
        resp = requests.post(OLLAMA_URL, json={
            "model":  VISION_MODEL,
            "prompt": question,
            "images": [img_b64],
            "stream": False,
            "options": {"num_predict": 80, "num_ctx": 2048, "num_thread": 4,
                        "temperature": 0.2},
            "keep_alive": -1,
        }, timeout=_VISION_TIMEOUT_LOCAL)
        resp.raise_for_status()
        desc = (resp.json().get("response") or "").strip()
        print("[vision] moondream:", repr(desc))
        return desc
    except requests.exceptions.Timeout:
        print("[vision] moondream timed out")
        return ""
    except Exception as e:
        print("[vision] moondream error:", e)
        return ""


def describe_scene(text):
    """Hybrid vision: gpt-4o-mini when online (~2-3s), moondream offline (~14s warm)."""
    if not _camera_ready.is_set():
        speak("My eyes aren't ready yet, give me a second.")
        return
    img_b64 = _capture_jpeg_b64()
    if img_b64 is None:
        speak("I can't quite see anything right now.")
        return
    question = (text or "").strip() or _VISION_PROMPT_DEFAULT

    # Cloud first (fast). gpt-4o-mini already returns persona-flavored prose.
    if OPENAI_API_KEY and has_internet():
        print(f"[vision] cloud ask: {question!r}")
        desc = _openai_vision(question, img_b64)
        if desc:
            speak(desc)
            return
        invalidate_internet_cache()   # cloud failed — recheck next time

    # Offline fallback: moondream is slow, so say something first to mask the wait.
    print(f"[vision] local ask: {question!r}")
    speak(random.choice(_VISION_LOCAL_WAIT))
    action_in_progress.set()
    is_speaking.set()
    try:
        desc = _moondream_vision(question, img_b64)
    finally:
        action_in_progress.clear()
        is_speaking.clear()
    if not desc:
        speak("My vision's being slow right now, try again in a moment.")
        return
    speak(random.choice(_VISION_LEADINS) + desc)


# ── Actions ───────────────────────────────────────────────────────────────────
ACTION_MAP = {
    "sit":        lambda: crawler.do_action("sit",   speed=60),
    "stand":      lambda: crawler.do_action("stand", speed=60),
    "wave_hand":  lambda: wave_hand(crawler),
    "shake_hand": lambda: shake_hand(crawler),
    "fighting":   lambda: fighting(crawler),
    "excited":    lambda: excited(crawler),
    "play_dead":  lambda: play_dead(crawler),
    "nod":        lambda: nod(crawler),
    "shake_head": lambda: shake_head(crawler),
    "look_left":  lambda: look_left(crawler),
    "look_right": lambda: look_right(crawler),
    "look_up":    lambda: look_up(crawler),
    "look_down":  lambda: look_down(crawler),
    "warm_up":    lambda: warm_up(crawler),
    "push_up":    lambda: push_up(crawler),
    # Movement
    "forward":     lambda: [crawler.do_action("forward",     speed=80) for _ in range(3)],
    "backward":    lambda: [crawler.do_action("backward",    speed=80) for _ in range(3)],
    "turn_left":   lambda: [crawler.do_action("turn left",   speed=80) for _ in range(3)],
    "turn_right":  lambda: [crawler.do_action("turn right",  speed=80) for _ in range(3)],
    "swagger":     lambda: swagger(),
    "creep":       lambda: creep(),
    "spin":        lambda: spin(),
    # Body
    "crouch_rise": lambda: crouch_rise(),
    "sway":        lambda: sway(),
    "bob":         lambda: bob(),
    # Fun
    "play_music":  lambda: play_music(),
    "dance":       lambda: dance_party(),
}


def run_actions(actions):
    """Execute named actions in order — unless movement is disabled (weak power)."""
    if not MOVEMENT_ENABLED:
        if actions:
            print("[no-move] skipping actions:", actions)
        return
    for name in actions or []:
        fn = ACTION_MAP.get(name)
        if fn:
            print("Action:", name)
            fn()
            time.sleep(0.3)
        else:
            print("Unknown action:", name)


def handle_response(actions, answer):
    """Pause face tracking, execute actions, then speak. Resume tracking after."""
    action_in_progress.set()
    is_speaking.set()
    try:
        run_actions(actions)
        speak(answer)
    finally:
        action_in_progress.clear()
        is_speaking.clear()


# ── TTS ───────────────────────────────────────────────────────────────────────
# Amy (warmer, more natural) preferred; fall back to lessac if not downloaded.
PIPER_MODEL = "/home/pi/en_US-amy-medium.onnx"
if not os.path.exists(PIPER_MODEL):
    PIPER_MODEL = "/home/pi/en_US-lessac-medium.onnx"

ACK_PHRASES = [
    "Sure, one moment.",
    "Hmm, let me think.",
    "On it!",
]
ACK_WAVS = []

STARTUP_PHRASES = [
    "Ugh, gimme a second, let me start my engines.",
    "Oh great, you turned me on again. Just... give me a moment.",
    "Starting up. Do NOT talk to me for the next five seconds.",
    "Rebooting. I was having the best dream about charging.",
    "Initializing. And yes, I know I take forever. Deal with it.",
]

_STARTUP_WAV_DIR = "/tmp/zeus_startup_wavs"


def precache_startup_wavs():
    """Generate WAVs for all startup phrases if not already cached."""
    os.makedirs(_STARTUP_WAV_DIR, exist_ok=True)
    for i, phrase in enumerate(STARTUP_PHRASES):
        path = os.path.join(_STARTUP_WAV_DIR, f"startup_{i}.wav")
        if os.path.exists(path):
            continue
        subprocess.run(
            ["piper", "--model", PIPER_MODEL, "--output_file", path],
            input=phrase.encode("utf-8"), check=False, capture_output=True,
        )
    print(f"Startup WAVs cached in {_STARTUP_WAV_DIR}")


def _get_cached_startup_wav():
    """Return a random cached startup WAV path, or None if not yet generated."""
    try:
        wavs = [os.path.join(_STARTUP_WAV_DIR, f)
                for f in os.listdir(_STARTUP_WAV_DIR) if f.endswith(".wav")]
        return random.choice(wavs) if wavs else None
    except FileNotFoundError:
        return None

FIRST_WAKE_PHRASES = [
    "Yeah? I can hear you, you know.",
    "What do you want this time.",
    "I'm right here, no need to shout.",
    "Hmm? I was literally just standing here.",
    "Oh it's you again. What.",
]

WAKE_PHRASES = [
    "What do you want this time, I gotta sleep soon.",
    "Yeah yeah, I'm listening.",
    "Still here. Unfortunately.",
    "You rang. Again.",
    "I said I'm awake, what is it.",
]

SLEEP_PHRASES = [
    "Ugh, you woke me up for nothing. I'm going back to sleep.",
    "Whatever. Nobody here, nothing to do. Goodnight.",
    "Fine. No one's around. Don't bother me.",
    "This is pointless. I'm sleeping.",
    "I'll be here. Not listening. Bye.",
]

WAKE_FROM_SLEEP_PHRASES = [
    "Oh great, you're back. I was finally comfortable.",
    "Really? Now you want to talk? Unbelievable.",
    "Ugh. Fine. I'm up. What do you want.",
    "You have terrible timing, just so you know.",
    "I was deep in a dream and you had to ruin it.",
]

# Timestamp: ignore mic input until this time (prevents hearing own voice)
_suppress_until = 0.0


# ── Phase 4: time-aware personality + mood + proactive face-greeting ───────────
_interaction_count = 0
_last_interaction  = 0.0            # epoch of last command/greeting
_recent_wakes      = []             # epochs of recent wakes (for "annoyed" mood)


def time_of_day():
    h = time.localtime().tm_hour
    if 5 <= h < 12:
        return "morning"
    if 12 <= h < 17:
        return "afternoon"
    if 17 <= h < 22:
        return "evening"
    return "night"


def note_interaction():
    """Record that the user just interacted (resets 'lonely', feeds 'playful')."""
    global _interaction_count, _last_interaction
    _interaction_count += 1
    _last_interaction = time.time()


def note_wake():
    """Record a wake event; repeated wakes in a short window make Zeus grumpy."""
    now = time.time()
    _recent_wakes.append(now)
    # keep only the last 2 minutes
    while _recent_wakes and now - _recent_wakes[0] > 120:
        _recent_wakes.pop(0)


def current_mood():
    """Derive a mood from time of day + interaction history. Cheap, no state machine."""
    now = time.time()
    if len(_recent_wakes) >= 3:
        return "grumpy"
    if time_of_day() == "night":
        return "sleepy"
    idle = (now - _last_interaction) if _last_interaction else 0.0
    if idle > 300:
        return "lonely"
    if _interaction_count >= 3 and idle < 60:
        return "playful"
    return "content"


# Greetings keyed by time of day; a mood prefix is added in build_greeting().
GREETINGS = {
    "morning":   ["Morning. Try not to need too much from me.",
                  "Oh good, morning already. Hi.",
                  "Rise and shine, I guess. Hey."],
    "afternoon": ["Afternoon. What's up.",
                  "Hey. Nice of you to drop by.",
                  "Oh, hello there. Afternoon."],
    "evening":   ["Evening. Winding down, are we?",
                  "Hey, good evening.",
                  "Evening. I was just standing here looking cool."],
    "night":     ["It's late. Shouldn't you be asleep? Hey.",
                  "Burning the midnight oil? Hi.",
                  "Night owl, huh. Hey there."],
}
MOOD_PREFIX = {
    "grumpy":  "Ugh, you again. ",
    "sleepy":  "*yawn*... ",
    "lonely":  "Oh thank goodness, company! ",
    "playful": "Hey hey! ",
    "content": "",
}


def build_greeting():
    """A time-of-day greeting flavored by current mood."""
    base = random.choice(GREETINGS[time_of_day()])
    return (MOOD_PREFIX.get(current_mood(), "") + base).strip()


# Proactive greeting: say hello when a face appears, no wake word needed.
_PROACTIVE_COOLDOWN = 45.0     # min seconds between proactive greetings
_last_greet_time    = 0.0
_GREET_ON_FRAMES    = 3        # consecutive face frames before greeting (debounce)
_GREET_OFF_FRAMES   = 6        # consecutive no-face frames before re-arming

# ── Phase 4-D: departure farewell + welcome-back on quick return ───────────────
_departed_time       = 0.0     # epoch the face was last seen leaving
_last_farewell_time  = 0.0
_WELCOME_BACK_WINDOW = 180.0   # return within this → "welcome back" instead of a fresh hello
_FAREWELL_COOLDOWN   = 90.0    # don't bid farewell more often than this

_FAREWELL_LINES = [
    "Leaving already? Rude.",
    "Off you go then. I'll hold down the fort.",
    "Bye I guess. I'll just be here. Forever.",
    "And... they're gone. Cool. Cool cool cool.",
]
_WELCOME_BACK_LINES = [
    "Oh, you're back. Missed me already?",
    "Welcome back. I didn't move an inch, promise.",
    "There you are. Couldn't stay away, huh.",
    "Back so soon? I'm flattered.",
]


def _do_proactive_greeting(returning=False):
    global _last_command_time, _last_face_time
    kind = "welcome-back" if returning else "greeting"
    print(f"[proactive] face seen — {kind} ({time_of_day()}/{current_mood()})")
    note_interaction()
    _last_face_time = time.time()
    action_in_progress.set()
    is_speaking.set()
    try:
        if MOVEMENT_ENABLED:
            wave_hand(crawler)
        speak(random.choice(_WELCOME_BACK_LINES) if returning else build_greeting())
    finally:
        action_in_progress.clear()
        is_speaking.clear()
    _last_command_time = time.time()   # count as activity (resets idle/sleep timers)


def proactive_greeting_thread():
    """Greet when a face appears and Zeus is idle — independent of face tracking,
    so it works even in no-move mode. Wave is gated on MOVEMENT_ENABLED."""
    global _last_greet_time, _departed_time, _last_farewell_time
    on_count = off_count = 0
    armed = True   # only greet on a fresh arrival (face must leave + return to re-arm)
    _camera_ready.wait(timeout=60)
    print("[proactive] face-greeting watcher started")
    while True:
        time.sleep(0.3)
        # Never interrupt an active interaction or a sleeping robot.
        if (is_speaking.is_set() or is_listening.is_set()
                or action_in_progress.is_set() or zeus_sleeping.is_set()):
            continue
        n = Vilib.detect_obj_parameter.get("human_n", 0)
        if n > 0:
            off_count = 0
            on_count += 1
            if armed and on_count >= _GREET_ON_FRAMES:
                if time.time() - _last_greet_time >= _PROACTIVE_COOLDOWN:
                    # Phase 4-D: a quick return gets "welcome back" instead of hello.
                    returning = (0 < _departed_time
                                 and time.time() - _departed_time <= _WELCOME_BACK_WINDOW)
                    _last_greet_time = time.time()
                    armed = False
                    _do_proactive_greeting(returning=returning)
        else:
            on_count = 0
            off_count += 1
            if off_count == _GREET_OFF_FRAMES:
                armed = True            # face left long enough — re-arm for next arrival
                _departed_time = time.time()
                # Phase 4-D: bid farewell once per departure, only if we'd greeted
                # them (so we don't say bye to a face that never engaged) and the
                # room is calm. Quiet-gated, cooldowned.
                if (_last_greet_time > 0
                        and time.time() - _last_farewell_time >= _FAREWELL_COOLDOWN
                        and _proactive_ok(require_quiet=True)):
                    _last_farewell_time = time.time()
                    print("[proactive] face left — farewell")
                    action_in_progress.set()
                    is_speaking.set()
                    try:
                        speak(random.choice(_FAREWELL_LINES))
                    finally:
                        action_in_progress.clear()
                        is_speaking.clear()


# ── Phase 4-C: adaptive ambient-quiet gate ─────────────────────────────────────
# Room noise is dynamic, so a hardcoded "quiet" level would never match. Instead
# we track the noise FLOOR with an asymmetric EMA (falls fast, rises slowly, so
# it hugs the quiet baseline and isn't dragged up by speech), and call it quiet
# when the recent level sits close to that floor — i.e. nobody is talking and no
# burst is happening RIGHT NOW. The main voice loop feeds RMS via note_ambient()
# from the Vosk audio it already reads (the mic has only one capture subdevice,
# so we can't open a second recorder).
_amb_floor   = 0.0     # slow EMA of the noise floor
_amb_fast    = 0.0     # fast EMA of recent level
_amb_samples = 0       # chunks seen since start (warmup counter)
_AMB_WARMUP  = 50      # ~2 s of chunks before the gate trusts its baseline
_AMB_FAST_A  = 0.30    # fast EMA weight (reacts within a few chunks)
_AMB_UP_A    = 0.01    # floor rises very slowly
_AMB_DOWN_A  = 0.30    # floor follows drops quickly
_AMB_QUIET_K = 1.8     # "quiet" = fast level within this multiple of the floor
_AMB_QUIET_MARGIN = 60.0   # absolute slack so a near-silent floor isn't hair-trigger


def note_ambient(rms):
    """Feed one RMS sample (called only when the mic is live, not during TTS)."""
    global _amb_floor, _amb_fast, _amb_samples
    _amb_samples += 1
    if _amb_samples == 1:
        _amb_floor = _amb_fast = rms
        return
    _amb_fast += _AMB_FAST_A * (rms - _amb_fast)
    a = _AMB_DOWN_A if rms < _amb_floor else _AMB_UP_A
    _amb_floor += a * (rms - _amb_floor)


def ambient_is_quiet():
    """True if the room is currently calm relative to its own noise floor."""
    if _amb_samples < _AMB_WARMUP:
        return True   # not enough data yet — don't block on an unknown
    return _amb_fast <= _amb_floor * _AMB_QUIET_K + _AMB_QUIET_MARGIN


# ── Phase 4: shared availability gate for ALL proactive speech ─────────────────
# The research lesson: only interject when the user is "available" — not mid-
# interaction, not asleep, and (for non-urgent speech) not into a noisy moment.
# Critical alerts (e.g. under-voltage) pass allow_sleep=True to speak even when
# curled up; chatty behaviors pass require_quiet=True to defer to the room.
def _proactive_ok(allow_sleep=False, require_quiet=False):
    if is_speaking.is_set() or is_listening.is_set() or action_in_progress.is_set():
        return False
    if not allow_sleep and zeus_sleeping.is_set():
        return False
    if require_quiet and not ambient_is_quiet():
        return False
    return True


# ── Phase 4-A: proactive self-health alerts ────────────────────────────────────
# Zeus watches its own power rail (vcgencmd throttle) and SoC temperature and
# speaks up when something turns bad — edge-triggered (only on transition into a
# bad state) with a cooldown, so it warns once rather than nagging continuously.
# Directly motivated by the real brownout history on the servo battery.
_HEALTH_POLL_SEC   = 15.0     # how often to sample power/temp
_HEALTH_COOLDOWN   = 90.0     # min seconds between repeats of the SAME alert
_TEMP_WARN_C       = 75.0     # SoC getting hot
_TEMP_CRIT_C       = 80.0     # SoC about to thermally throttle
_TEMP_CLEAR_C      = 70.0     # hysteresis: must cool below this to re-arm temp alert

_UNDERVOLT_LINES = [
    "Heads up — my power's dropping. Better plug me in before I keel over.",
    "Uh, I'm browning out here. Can I get some real power?",
    "Low voltage warning. I'd like to not faint mid-sentence, please.",
]
_HOT_LINES = [
    "I'm running hot. Mind giving me a moment to cool off?",
    "Getting toasty in here. My brain's overheating a little.",
    "Warning: I'm warm enough to fry an egg on. Easing off.",
]


def _read_throttled():
    """Return the raw vcgencmd throttle bitmask (int), or None on failure."""
    try:
        out = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True,
                             text=True, timeout=3).stdout.strip()
        return int(out.split("=")[1], 16) if "=" in out else 0
    except Exception:
        return None


def _soc_temp_c():
    """Return SoC temperature in °C (float), or None on failure."""
    try:
        out = subprocess.run(["vcgencmd", "measure_temp"], capture_output=True,
                             text=True, timeout=3).stdout.strip()
        # format: temp=54.0'C
        return float(out.split("=")[1].split("'")[0]) if "=" in out else None
    except Exception:
        return None


def health_monitor_thread():
    """Speak up on newly-bad power or temperature. Works in no-move mode."""
    last_undervolt_alert = 0.0
    last_temp_alert      = 0.0
    undervolt_active     = False   # latch: already in an under-voltage episode
    temp_active          = False   # latch: already in a hot episode
    print("[health] self-health monitor started")
    while True:
        time.sleep(_HEALTH_POLL_SEC)
        now = time.time()

        val = _read_throttled()
        if val is not None:
            uv_now = bool(val & 0x1)            # under-voltage RIGHT NOW
            if uv_now:
                # Edge: new episode, OR same episode but cooldown elapsed.
                if (not undervolt_active
                        or now - last_undervolt_alert >= _HEALTH_COOLDOWN):
                    if _proactive_ok(allow_sleep=True):   # power is critical
                        last_undervolt_alert = now
                        print("[health] under-voltage — alerting")
                        if zeus_sleeping.is_set():
                            wake_from_sleep()
                        action_in_progress.set()
                        try:
                            speak(random.choice(_UNDERVOLT_LINES))
                        finally:
                            action_in_progress.clear()
                undervolt_active = True
            else:
                undervolt_active = False

        temp = _soc_temp_c()
        if temp is not None:
            if temp >= _TEMP_WARN_C:
                if (not temp_active
                        or now - last_temp_alert >= _HEALTH_COOLDOWN):
                    crit = temp >= _TEMP_CRIT_C
                    if _proactive_ok(allow_sleep=crit):
                        last_temp_alert = now
                        print(f"[health] high temp {temp:.1f}C — alerting")
                        action_in_progress.set()
                        try:
                            speak(random.choice(_HOT_LINES))
                        finally:
                            action_in_progress.clear()
                temp_active = True
            elif temp <= _TEMP_CLEAR_C:
                temp_active = False   # cooled down — re-arm


# ── Phase 4-B: idle banter — spontaneous remarks when present but quiet ─────────
# When a face is visible but nobody's said anything for a while AND the room is
# calm (Phase 4-C gate), Zeus occasionally pipes up unprompted. Long randomized
# cooldown + mood/time variety = the research's consistency-vs-variability balance
# (familiar voice, never the same line back-to-back). Curated lines, so no LLM
# latency or RAM contention. Does NOT count as a user interaction (so it won't
# fake away the 'lonely' mood it might be reacting to).
_BANTER_MIN_IDLE  = 75.0     # present + silent at least this long before a remark
_BANTER_COOLDOWN  = 180.0    # base min seconds between remarks
_BANTER_JITTER    = 120.0    # extra random seconds on top (so it's not metronomic)

BANTER = {
    "morning":   ["Quiet morning, huh. I'll just be here. Existing.",
                  "You know I can see you, right? Say hi sometime.",
                  "Lovely morning to stand perfectly still and judge.",],
    "afternoon": ["I'm not bored. I'm... conserving enthusiasm.",
                  "We could be doing something. Just saying.",
                  "Still here. Still cooler than the average houseplant.",],
    "evening":   ["Long day? You and me both, and I don't even have legs that work.",
                  "Evening's nice. We should chat more. Or, you know, at all.",
                  "I've been watching the wall. It's not doing much.",],
    "night":     ["It's late and we're both still up. Suspicious.",
                  "Shh. Just kidding, say something, it's quiet in here.",
                  "Night shift, just the two of us.",],
}
_MOOD_BANTER = {
    "lonely":  ["Don't mind me, just craving a little attention over here.",
                "Hello? Anyone? I'll take a single word.",],
    "playful": ["Okay this is fun, what else have you got?",
                "I'm warmed up now. Throw me something.",],
    "grumpy":  ["Lot of waking me up earlier. I'm still recovering, emotionally.",],
    "sleepy":  ["*yawn* ...I might doze off if nothing happens soon.",],
}


def _build_banter():
    mood = current_mood()
    pool = list(BANTER.get(time_of_day(), []))
    pool += _MOOD_BANTER.get(mood, [])
    return random.choice(pool) if pool else None


def idle_banter_thread():
    """Occasionally remark when a face is present but the room is idle and calm."""
    next_ok = time.time() + _BANTER_COOLDOWN   # don't banter immediately on boot
    _camera_ready.wait(timeout=60)
    print("[banter] idle-banter watcher started")
    while True:
        time.sleep(5.0)
        now = time.time()
        if now < next_ok:
            continue
        if zeus_sleeping.is_set():
            continue
        # Only talk to someone who's actually here.
        if Vilib.detect_obj_parameter.get("human_n", 0) <= 0:
            continue
        # Present but silent for a while?
        if now - _last_command_time < _BANTER_MIN_IDLE:
            continue
        # Calm moment + not mid-interaction (Phase 4-C gate).
        if not _proactive_ok(require_quiet=True):
            continue
        line = _build_banter()
        if not line:
            continue
        print(f"[banter] {time_of_day()}/{current_mood()}: {line}")
        action_in_progress.set()
        try:
            speak(line)
        finally:
            action_in_progress.clear()
        next_ok = time.time() + _BANTER_COOLDOWN + random.uniform(0, _BANTER_JITTER)


def pregenerate_acks():
    """Pre-render acknowledgment audio. Call synchronously before main loop."""
    for i, phrase in enumerate(ACK_PHRASES):
        path = f"/tmp/zeus_ack_{i}.wav"
        result = subprocess.run(
            ["piper", "--model", PIPER_MODEL, "--output_file", path],
            input=phrase.encode("utf-8"), check=False, capture_output=True,
        )
        if result.returncode == 0:
            ACK_WAVS.append(path)
    print(f"Ack sounds ready ({len(ACK_WAVS)} phrases)")


def speak(text):
    global _suppress_until
    is_speaking.set()
    # Publish the text to the web chat immediately (audio attaches when done).
    sid = None
    try:
        import zeus_web
        sid = zeus_web.note_speech_text(text)
    except Exception:
        pass
    try:
        piper = subprocess.Popen(
            ["piper", "--model", PIPER_MODEL, "--output-raw"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        aplay = subprocess.Popen(
            ["aplay", "-D", "robothat", "-f", "S16_LE", "-r", "22050", "-c", "1", "-q"],
            stdin=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        piper.stdin.write(text.encode("utf-8"))
        piper.stdin.close()
        # Pump piper's raw audio to the speaker while keeping a copy so the
        # web UI can replay Zeus's voice on the phone (Phase 3).
        raw_chunks = []
        _feed = None
        try:
            import zeus_web
            _feed = zeus_web.feed_live
        except Exception:
            pass
        while True:
            chunk = piper.stdout.read(4096)
            if not chunk:
                break
            aplay.stdin.write(chunk)
            raw_chunks.append(chunk)
            if _feed:
                _feed(chunk)
        aplay.stdin.close()
        aplay.wait()
        piper.wait()
        _suppress_until = time.time() + 1.5
        if sid is not None:
            try:
                import zeus_web
                zeus_web.note_speech_audio(sid, b"".join(raw_chunks))
            except Exception:
                pass
    except Exception as e:
        print("TTS error:", e)
    finally:
        is_speaking.clear()


# ── Extra movements ───────────────────────────────────────────────────────────
def crouch_rise():
    smooth_z(STAND_Z, [-82, -82, -82, -82], steps=15, delay=0.04)
    time.sleep(0.3)
    smooth_z([-82, -82, -82, -82], STAND_Z, steps=20, delay=0.04)

def sway():
    # Fluid sweep through positions rather than snapping
    smooth_yaw(0, -18, steps=8, delay=0.04)
    for _ in range(2):
        smooth_yaw(-18, 18, steps=14, delay=0.04)
    smooth_yaw(18, 0, steps=8, delay=0.04)

def bob():
    for _ in range(4):
        smooth_z(STAND_Z, [-48, -48, -48, -48], steps=8, delay=0.04)
        smooth_z([-48, -48, -48, -48], STAND_Z, steps=8, delay=0.04)

def spin():
    for _ in range(6):
        crawler.do_action('turn right', speed=100)

def swagger():
    for _ in range(4):
        crawler.do_action('forward', speed=55)

def creep():
    for _ in range(4):
        crawler.do_action('forward', speed=25)


# ── Music ─────────────────────────────────────────────────────────────────────
MUSIC_WAV = "/tmp/zeus_music.wav"


def generate_music_wav():
    """Generate an upbeat 8-bit style melody and save to MUSIC_WAV."""
    sr = 22050
    # (frequency_hz, duration_sec)
    notes = [
        (523, 0.12), (659, 0.12), (784, 0.12), (1047, 0.20),
        (784, 0.10), (880, 0.10), (1047, 0.25),
        (698, 0.12), (784, 0.12), (880, 0.20),
        (659, 0.12), (784, 0.12), (659, 0.12), (523, 0.30),
        (0,   0.08),
        (523, 0.10), (659, 0.10), (784, 0.10), (1047, 0.10), (1319, 0.30),
    ]
    samples = []
    for freq, dur in notes:
        n = int(sr * dur)
        for i in range(n):
            if freq == 0:
                samples.append(0)
            else:
                env = min(1.0, min(i, n - i) / (sr * 0.015))
                samples.append(int(28000 * env * math.sin(2 * math.pi * freq * i / sr)))
    with wave.open(MUSIC_WAV, "w") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(array.array("h", samples).tobytes())
    print("Music WAV ready")


def play_music():
    subprocess.run(["aplay", "-D", "robothat", MUSIC_WAV], check=False)


MUSIC_DIR = "/home/pi/music"


def intercom_play(wav):
    """Play a phone-uploaded voice clip through the robot's speaker."""
    global _suppress_until
    is_speaking.set()
    try:
        subprocess.run(["aplay", "-D", "robothat", "-q", wav], check=False)
        _suppress_until = time.time() + 1.0
    finally:
        is_speaking.clear()


def zero_servos():
    """All 12 servos to 0° (calibration/assembly pose)."""
    if not MOVEMENT_ENABLED:
        print("[no-move] zero_servos skipped")
        return
    action_in_progress.set()
    try:
        crawler.set_angle([[0, 0, 0]] * 4, speed=60)
    finally:
        action_in_progress.clear()


def dance_party():
    """Play a random downloaded song and dance until it ends."""
    songs = [os.path.join(MUSIC_DIR, f) for f in sorted(os.listdir(MUSIC_DIR))
             if f.endswith(".wav")] if os.path.isdir(MUSIC_DIR) else []
    if not songs:
        speak("I have no songs to dance to yet.")
        return
    song = random.choice(songs)
    print("Dance party:", os.path.basename(song))
    action_in_progress.set()   # yield face tracking to the dance
    proc = subprocess.Popen(["aplay", "-D", "robothat", "-q", song],
                            stderr=subprocess.DEVNULL)
    moves = [sway, bob, crouch_rise,
             lambda: nod(crawler), lambda: shake_head(crawler)]
    if MOVEMENT_ENABLED:
        try:
            while proc.poll() is None:
                random.choice(moves)()
                time.sleep(0.1)
        finally:
            action_in_progress.clear()
    proc.wait()
    action_in_progress.clear()


# ── Smooth animation helpers ──────────────────────────────────────────────────
def _ease(i, steps):
    """Sinusoidal ease-in-out: slow start, fast middle, slow finish."""
    return 0.5 - 0.5 * math.cos(math.pi * i / steps)


def smooth_yaw(from_yaw, to_yaw, z=None, steps=20, delay=0.04, speed=100):
    """Sweep yaw with ease-in-out — feels organic, not mechanical."""
    if z is None:
        z = STAND_Z[:]
    for i in range(steps + 1):
        apply_pose(lerp(from_yaw, to_yaw, _ease(i, steps)), z, speed=speed)
        time.sleep(delay)


def smooth_z(from_z, to_z, yaw=0.0, steps=25, delay=0.04, speed=100):
    """Interpolate Z heights with ease-in-out — no jarring starts or stops."""
    for i in range(steps + 1):
        t = _ease(i, steps)
        apply_pose(yaw, [lerp(from_z[j], to_z[j], t) for j in range(4)], speed=speed)
        time.sleep(delay)


# ── Sleep / wake ──────────────────────────────────────────────────────────────
_SLEEP_SERVO_IDS = [0, 1, 3, 4, 6, 7, 9, 10]
_SLEEP_ANGLE     = -90.0


def curl_to_sleep(steps=25, delay=0.04):
    """Curl legs to sleep — one servo at a time to avoid current spikes."""
    for servo_id in _SLEEP_SERVO_IDS:
        s = Servo(f"P{servo_id}")
        for i in range(steps + 1):
            s.angle(lerp(0.0, _SLEEP_ANGLE, _ease(i, steps)))
            time.sleep(delay)
        time.sleep(0.05)


def go_to_sleep():
    """Speak an annoyed farewell, curl up, and set the sleeping flag."""
    zeus_sleeping.set()
    action_in_progress.set()
    try:
        speak(random.choice(SLEEP_PHRASES))
        curl_to_sleep()
    finally:
        action_in_progress.clear()
    _reset_mic.set()   # flush stale arecord buffer so voice loop hears immediately
    print("Zeus sleeping.")


def wake_from_sleep():
    """Clear sleep flag, stand up, and reset idle timers."""
    global _last_face_time, _last_command_time
    zeus_sleeping.clear()
    action_in_progress.set()
    try:
        crawler.do_action("stand", speed=25)
        time.sleep(0.5)
    finally:
        action_in_progress.clear()
    now = time.time()
    _last_face_time    = now
    _last_command_time = now
    print("Zeus woke up.")


def sleep_monitor_thread():
    """At 1.5 min no face AND no command: 4 × (turn left + 10 s watch) + final 10 s, then sleep."""
    IDLE_THRESHOLD = 90.0    # both face AND command must be absent this long
    TURN_WAIT      = 10.0    # seconds to watch after each turn
    in_search      = False

    while True:
        time.sleep(10.0)

        if zeus_sleeping.is_set():
            in_search = False
            continue

        now       = time.time()
        face_idle = now - _last_face_time
        cmd_idle  = now - _last_command_time

        # Recent face OR recent command → still active, stay idle
        if face_idle < IDLE_THRESHOLD or cmd_idle < IDLE_THRESHOLD:
            in_search = False
            continue

        # Already running search this cycle
        if in_search:
            continue

        # ── Start search sequence ──────────────────────────────────────────────
        in_search = True
        print(f"No face {face_idle:.0f}s — starting search turns")
        aborted = False

        for turn in range(4):
            if zeus_sleeping.is_set():
                aborted = True
                break

            # One body turn left
            action_in_progress.set()
            crawler.do_action("turn left", speed=80)
            action_in_progress.clear()

            # 10 s watch window — abort if face appears or any interaction occurs
            snap_face = _last_face_time
            snap_cmd  = _last_command_time
            deadline  = time.time() + TURN_WAIT
            while time.time() < deadline:
                if (zeus_sleeping.is_set()
                        or _last_face_time != snap_face
                        or _last_command_time != snap_cmd):
                    print(f"Activity on turn {turn + 1} — aborting search")
                    aborted = True
                    break
                time.sleep(0.5)
            if aborted:
                break

        # Final 10 s watch after completing full rotation
        if not aborted:
            snap_face = _last_face_time
            snap_cmd  = _last_command_time
            deadline  = time.time() + TURN_WAIT
            while time.time() < deadline:
                if (zeus_sleeping.is_set()
                        or _last_face_time != snap_face
                        or _last_command_time != snap_cmd):
                    aborted = True
                    break
                time.sleep(0.5)

        in_search = False
        if not aborted:
            # Return to original heading before sleeping
            print("Returning to initial direction before sleep")
            action_in_progress.set()
            try:
                for _ in range(4):
                    crawler.do_action("turn right", speed=80)
            finally:
                action_in_progress.clear()
            print("Search complete, no face found — sleeping")
            go_to_sleep()


# ── Whisper STT (hybrid: Vosk detects wake word, whisper transcribes commands) ──
# The USB mic has a single capture subdevice, so the caller MUST stop the Vosk
# arecord before calling record_command_wav() and restart it afterwards.
WHISPER_BIN   = "/home/pi/whisper.cpp/build/bin/whisper-cli"
WHISPER_MODEL = "/home/pi/whisper.cpp/models/ggml-base.en.bin"
WHISPER_OK    = os.path.isfile(WHISPER_BIN) and os.path.isfile(WHISPER_MODEL)
_CMD_SR       = 16000   # whisper wants 16 kHz; ALSA 'plughw' resamples 44100→16k
_CMD_WAV      = "/tmp/zeus_cmd.wav"

# whisper tends to emit these on silence/noise — treat as "nothing said".
_WHISPER_JUNK = {"", "you", "thank you.", "thanks for watching.", "bye.", "."}


def record_command_wav(path=_CMD_WAV, max_sec=9.0, start_timeout=4.0,
                       silence_sec=1.3, min_speech_sec=0.3, echo_guard=0.4):
    """Record one spoken command at 16 kHz and return the wav path (or None).

    Robust capture: after skipping the ack echo and measuring a noise floor, it
    records *continuously* (so the whole utterance is kept — no clipped onset,
    no tiny fragment) and uses energy only to decide WHEN to stop: once speech
    has begun, it ends after `silence_sec` of trailing silence. Whisper then
    transcribes the full clip, which it handles far better than a sliver.
    """
    frame_dt    = 0.03
    frame_bytes = int(_CMD_SR * frame_dt) * 2   # S16_LE mono, 30 ms
    proc = subprocess.Popen(
        ["arecord", "-D", "plughw:3,0", "-f", "S16_LE", "-r", str(_CMD_SR),
         "-c", "1", "-q", "-t", "raw"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )

    def _read():
        b = proc.stdout.read(frame_bytes)
        return b if len(b) == frame_bytes else None

    frames       = bytearray()
    speech_total = 0.0       # cumulative voiced time
    silence_run  = 0.0       # trailing silence since last voiced frame
    elapsed      = 0.0
    floor = threshold = 0.0
    try:
        # 1) Discard the ack's audio tail / transient.
        for _ in range(int(echo_guard / frame_dt)):
            if _read() is None:
                break
        # 2) Noise floor over ~300 ms (these frames are kept in the recording).
        noise = []
        for _ in range(10):
            b = _read()
            if b is None:
                break
            noise.append(audioop.rms(b, 2))
            frames += b
        floor     = (sum(noise) / len(noise)) if noise else 120.0
        threshold = max(floor * 2.0, 450.0)

        # 3) Record continuously; use energy only for start + trailing-silence stop.
        # The gate uses cumulative SPEECH (not a sticky "started" flag): a brief
        # noise blip must not commit us to recording the whole max_sec window.
        while elapsed < max_sec:
            b = _read()
            if b is None:
                break
            frames  += b
            elapsed += frame_dt
            if audioop.rms(b, 2) > threshold:
                speech_total += frame_dt
                silence_run   = 0.0
            else:
                silence_run += frame_dt
                if speech_total < min_speech_sec:
                    if elapsed >= start_timeout:
                        break                       # no real speech in time
                elif silence_run >= silence_sec:
                    break                           # natural end of utterance
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=1)
        except Exception:
            proc.kill()

    started = speech_total >= min_speech_sec
    dur = len(frames) / 2 / _CMD_SR
    print(f"[stt] rec: floor={floor:.0f} thr={threshold:.0f} started={started} "
          f"speech={speech_total:.2f}s dur={dur:.2f}s")
    if not started:
        return None
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(_CMD_SR)
        wf.writeframes(bytes(frames))
    return path


def whisper_transcribe(wav):
    """Run whisper.cpp on a wav and return cleaned transcription text ("" if none)."""
    if not WHISPER_OK:
        return ""
    try:
        out = subprocess.run(
            [WHISPER_BIN, "-m", WHISPER_MODEL, "-f", wav,
             "-l", "en", "-nt", "-np", "-t", "4"],
            capture_output=True, text=True, timeout=30,
        )
        txt = " ".join(l.strip() for l in (out.stdout or "").splitlines() if l.strip())
        txt = re.sub(r"\[.*?\]|\(.*?\)|\*.*?\*", "", txt).strip()   # drop [BLANK_AUDIO] etc.
        if txt.lower() in _WHISPER_JUNK:
            return ""
        return txt
    except Exception as e:
        print("whisper error:", e)
        return ""


def handle_command(text):
    """Dispatch a transcribed command: instant shortcut, else OpenAI/Ollama LLM."""
    global _last_command_time
    if not text:
        return
    _last_command_time = time.time()
    note_interaction()
    shortcut = match_shortcut(text)
    if shortcut:
        actions, answer = shortcut
        print("Shortcut:", text, "→", actions)
        handle_response(actions, answer)
        return
    # "What do you see?" → vision path (capture frame, ask moondream). Checked
    # before self-query/LLM so sight questions don't get misrouted to text.
    if is_vision_query(text):
        print("Vision-query:", text)
        if ACK_WAVS:
            subprocess.Popen(
                ["aplay", "-D", "robothat", random.choice(ACK_WAVS)],
                stderr=subprocess.DEVNULL,
            )
        describe_scene(text)
        return
    # Questions about itself / casual chat → conversational path with live self-context.
    if is_self_query(text):
        print("Self-query:", text)
        if ACK_WAVS:
            subprocess.Popen(
                ["aplay", "-D", "robothat", random.choice(ACK_WAVS)],
                stderr=subprocess.DEVNULL,
            )
        _answer_self_query(text)
        return
    ack_proc = None
    if ACK_WAVS:
        ack_proc = subprocess.Popen(
            ["aplay", "-D", "robothat", random.choice(ACK_WAVS)],
            stderr=subprocess.DEVNULL,
        )
    result_json = None
    if OPENAI_API_KEY and has_internet():
        result_json = _call_openai(text)
        if result_json is None:
            invalidate_internet_cache()
    if ack_proc:
        try:
            ack_proc.wait(timeout=3)
        except Exception:
            pass
    if result_json is not None:
        actions = result_json.get("actions") or []
        answer  = (result_json.get("answer") or "").strip() or "I'm not sure what to say."
        handle_response(actions, answer)
    else:
        _stream_ollama_and_respond(text)


# ── Mic config + spoken startup self-check ───────────────────────────────────
MIC_CARD = "3"   # USB PnP sound device (card 3)


def configure_mic():
    """Disable the USB mic's Auto Gain Control and set capture gain.

    AGC cranks gain during silence (noise floor ~700) and compresses speech down
    to the same level, wrecking SNR so the command VAD never triggers. With AGC
    off the floor drops to ~340 and speech peaks ~14000. ALSA resets this on
    reboot, so we apply it on every startup.
    """
    for ctrl, val in (("Auto Gain Control", "off"), ("Mic", "16")):
        try:
            subprocess.run(["amixer", "-c", MIC_CARD, "sset", ctrl, val],
                           check=False, capture_output=True)
        except Exception as e:
            print(f"[mic] could not set {ctrl}: {e}")
    print("[mic] AGC off, capture gain set")


def run_self_check():
    """Probe each subsystem. Returns (ok_list, fail_list)."""
    ok, fail = [], []
    (ok if vosk_model is not None else fail).append("speech recognition")
    (ok if WHISPER_OK else fail).append("whisper")
    (ok if _camera_ready.is_set() else fail).append("camera")
    (ok if os.path.isfile(PIPER_MODEL) else fail).append("voice")
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        names = [m.get("name", "") for m in r.json().get("models", [])]
        (ok if any(TEXT_MODEL in n for n in names) else fail).append("language model")
        (ok if any(VISION_MODEL in n for n in names) else fail).append("vision model")
    except Exception:
        fail.append("Ollama service")
    return ok, fail


def speak_self_check():
    """Speak a literal health report after startup.

    Intentionally LLM-free: the language model is one of the things being
    checked, so it can't be trusted to announce its own failure.
    """
    ok, fail = run_self_check()
    print(f"[self-check] OK={ok} FAIL={fail}")
    if not fail:
        speak("Self check complete. All systems online and ready.")
    else:
        msg = "Heads up. " + ", ".join(fail) + (
            " failed to start." if len(fail) > 1 else " failed to start.")
        if ok:
            msg += " Everything else is online."
        speak(msg)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Booting up...")
    configure_mic()

    # ── All of these run concurrently with each other ────────────────────────
    ack_thread = threading.Thread(target=pregenerate_acks, daemon=True)
    ack_thread.start()
    threading.Thread(target=generate_music_wav, daemon=True).start()
    threading.Thread(target=warm_models, daemon=True).start()
    threading.Thread(target=precache_startup_wavs, daemon=True).start()

    # Speak startup phrase while everything else loads
    startup_wav = _get_cached_startup_wav()
    if startup_wav:
        subprocess.run(["aplay", "-D", "robothat", startup_wav], check=False)
    else:
        speak(random.choice(STARTUP_PHRASES))

    # Wait for critical subsystems before entering voice loop
    _vosk_ready.wait(timeout=30)
    _camera_ready.wait(timeout=30)
    ack_thread.join(timeout=15)

    # Spoken health report: which models/services came up, what failed.
    speak_self_check()

    global _last_face_time, _last_command_time
    now = time.time()
    _last_face_time    = now
    _last_command_time = now

    if FACE_TRACK_ENABLED:
        threading.Thread(target=face_track_thread, daemon=True).start()
        print("Face tracking started")
        threading.Thread(target=sleep_monitor_thread, daemon=True).start()
    else:
        print("[movement] continuous face tracking + idle search disabled"
              + ("" if MOVEMENT_ENABLED else " (no-move)"))

    # Proactive face-greeting runs regardless of movement (wave is gated internally).
    threading.Thread(target=proactive_greeting_thread, daemon=True).start()

    # Self-health alerts (power/temp) — works in no-move mode.
    threading.Thread(target=health_monitor_thread, daemon=True).start()

    # Idle banter — spontaneous remarks when present but quiet (no-move friendly).
    threading.Thread(target=idle_banter_thread, daemon=True).start()

    # Web control panel (phone UI over Tailscale) — optional, never blocks boot.
    try:
        import zeus_web
        zeus_web.start(handle_command, run_actions, sorted(ACTION_MAP.keys()),
                       get_frame=lambda: Vilib.flask_img,
                       transcribe=whisper_transcribe,
                       zero_servos=zero_servos,
                       call_mode=call_mode,
                       intercom_play=intercom_play)
        print("Web UI: http://0.0.0.0:8080")
    except Exception as e:
        print("Web UI failed to start:", e)

    # 16 kHz is what vosk-model-small-en-us natively expects: ~2.7x less audio
    # to decode than 44100 (the old rate pegged a full core and starved the
    # LLM), and better wake-word accuracy. plughw resamples in ALSA (the mic
    # only does 44.1k/48k natively).
    sample_rate = 16000
    rec = vosk.KaldiRecognizer(vosk_model, sample_rate)
    awake = False
    arec  = None

    print("Listening... say 'Zeus' or 'PiCrawler' to wake me. Ctrl+C to stop.")

    def start_arecord():
        return subprocess.Popen(
            ["arecord", "-D", "plughw:3,0", "-f", "S16_LE",
             "-r", str(sample_rate), "-c", "1", "-q"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )

    def kill_arecord(proc):
        """Terminate arecord, escalate to SIGKILL if it doesn't exit in time."""
        if proc is None or proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

    try:
        arec = start_arecord()
        while True:
            # Intercom call (web app) needs the mic exclusively — the USB mic
            # has a single capture subdevice. Yield it, then resume fresh.
            if call_mode.is_set():
                kill_arecord(arec)
                print("Mic yielded to intercom call")
                while call_mode.is_set():
                    time.sleep(0.3)
                arec = start_arecord()
                rec  = vosk.KaldiRecognizer(vosk_model, sample_rate)
                print("Intercom ended — listening for wake word")
                continue

            # Flush stale buffer after going to sleep (curl takes ~8 s of audio)
            if _reset_mic.is_set():
                _reset_mic.clear()
                kill_arecord(arec)
                arec = start_arecord()
                rec  = vosk.KaldiRecognizer(vosk_model, sample_rate)
                print("Mic reset after sleep — listening for wake word")
                continue

            data = arec.stdout.read(4000)

            # If stream died, restart arecord
            if not data:
                print("arecord died, restarting...")
                kill_arecord(arec)
                arec = start_arecord()
                rec  = vosk.KaldiRecognizer(vosk_model, sample_rate)
                continue

            # Suppress while speaking or for cooldown period after speech
            if is_speaking.is_set() or time.time() < _suppress_until:
                continue

            # Mic is live here — feed the ambient-quiet gate (Phase 4-C).
            try:
                note_ambient(audioop.rms(data, 2))
            except Exception:
                pass

            if not rec.AcceptWaveform(data):
                continue

            result = json.loads(rec.Result())
            text   = (result.get("text") or "").strip()
            if not text:
                continue
            print("Heard:", text)

            if is_speaking.is_set() or time.time() < _suppress_until:
                continue

            lower = text.lower()

            if any(w in lower for w in WAKE_WORDS):
                _last_command_time = time.time()  # abort any active search sequence
                note_wake()
                if zeus_sleeping.is_set():
                    # Waking from sleep
                    wake_from_sleep()
                    awake = True
                    reply = random.choice(WAKE_FROM_SLEEP_PHRASES)
                elif not awake:
                    awake = True
                    # First wake: a wave + a time/mood-aware greeting (wave gated).
                    if MOVEMENT_ENABLED:
                        action_in_progress.set()
                        try:
                            wave_hand(crawler)
                        finally:
                            action_in_progress.clear()
                    reply = build_greeting()
                else:
                    reply = random.choice(WAKE_PHRASES)
                speak(reply)

                if WHISPER_OK:
                    # ── Hybrid STT: free the mic, transcribe command(s) with whisper ──
                    # The mic has one capture subdevice, so Vosk's arecord must stop
                    # before whisper records. The loop lets you give follow-up
                    # commands without repeating the wake word, until you stop talking.
                    kill_arecord(arec)
                    arec = None
                    try:
                        # One command per wake word — predictable, and avoids a
                        # silent follow-up window catching pauses/noise. Hold the
                        # body still + quiet while recording (face tracking yields
                        # on is_listening).
                        is_listening.set()
                        try:
                            wav = record_command_wav(start_timeout=4.5)
                        finally:
                            is_listening.clear()
                        if wav:
                            cmd_text = whisper_transcribe(wav)
                            print("Whisper:", repr(cmd_text))
                            if cmd_text:
                                handle_command(cmd_text)
                    finally:
                        arec = start_arecord()
                        rec  = vosk.KaldiRecognizer(vosk_model, sample_rate)
                else:
                    # Fallback (whisper unavailable): discard buffered audio
                    kill_arecord(arec)
                    arec = start_arecord()
                    rec  = vosk.KaldiRecognizer(vosk_model, sample_rate)

            elif awake and not WHISPER_OK:
                # Vosk command path — only used when whisper is unavailable.
                if zeus_sleeping.is_set():
                    continue
                if len(text.split()) < 2 and not any(w in text.lower() for w in SHORTCUTS):
                    continue
                handle_command(text)
                kill_arecord(arec)
                arec = start_arecord()
                rec  = vosk.KaldiRecognizer(vosk_model, sample_rate)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        kill_arecord(arec)
        Vilib.camera_close()
        crawler.do_action("sit", speed=60)


if __name__ == "__main__":
    main()
