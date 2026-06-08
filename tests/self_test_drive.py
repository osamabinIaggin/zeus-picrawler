#!/usr/bin/env python3
"""Autonomous end-to-end self-test: drives the REAL zeus pipeline without a human.

Run with zeus.service STOPPED (this process becomes the sole hardware owner).
It imports the live zeus module and feeds commands straight into handle_command(),
exercising: spoken self-check, ack, shortcut actions (real servo movement),
an LLM command, and every new self-aware/introspection query — all out the speaker.

The only thing it can't do is trip the Vosk wake word (synthetic audio is unreliable
for that); everything downstream of transcription is exercised for real.
"""
import time
import zeus

print("=== waiting for subsystems (vosk + camera) ===", flush=True)
zeus._vosk_ready.wait(timeout=40)
zeus._camera_ready.wait(timeout=40)
print(f"vosk_ready={zeus._vosk_ready.is_set()} camera_ready={zeus._camera_ready.is_set()}",
      flush=True)

print("\n=== generating ack clips ===", flush=True)
zeus.pregenerate_acks()
print(f"ACK_WAVS count = {len(zeus.ACK_WAVS)}", flush=True)

print("\n=== spoken self-check (startup health report) ===", flush=True)
zeus.speak_self_check()

# (label, command-text) — light on heavy movement to respect the battery/brownout.
BATTERY = [
    ("ACTION shortcut",   "wave"),
    ("ACTION shortcut",   "nod"),
    ("LLM command",       "do something excited"),
    ("SELF identity",     "what are you?"),
    ("SELF hearing",      "how do you hear me?"),
    ("SELF memory",       "how much memory do you have right now?"),
    ("SELF abilities",    "what can you do?"),
    ("SELF status",       "are you running okay?"),
    ("SELF power",        "have you had any power problems lately?"),
    ("SELF codebase",     "how does your face tracking work?"),
    ("CHAT casual",       "how are you feeling, buddy?"),
]

for label, text in BATTERY:
    print(f"\n########## [{label}] -> {text!r}", flush=True)
    t0 = time.time()
    try:
        zeus.handle_command(text)
        print(f"########## ok ({time.time()-t0:.1f}s)", flush=True)
    except Exception as e:
        print(f"########## FAILED: {e!r}", flush=True)
    time.sleep(1.2)

print("\n=== returning to sit ===", flush=True)
try:
    zeus.crawler.do_action("sit", speed=50)
except Exception as e:
    print("sit failed:", e, flush=True)

# Release the camera so the service can re-acquire it after this harness exits.
# (Vilib leaves camera/Flask workers holding /dev/video* otherwise.)
try:
    zeus.Vilib.camera_close()
    print("camera released", flush=True)
except Exception as e:
    print("camera_close failed:", e, flush=True)
print("=== self-test complete ===", flush=True)
